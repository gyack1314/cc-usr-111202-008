"""按预先指定优先级比较结局的分层胜比分析。

每一层内枚举治疗组与对照组的全部跨组配对，先比较生存，再比较无进展，
最后比较生活质量变化。时间结局只用已观察事件下结论：较早的已观察事件
为劣势；两个已观察事件同刻为确定平局；事件与删失同刻、删失先发生或
两个删失都为未知，未知时整对不可比较，不向低优先级下移。

方差按二样本 U 统计量的一阶投影计算：治疗行均值向量的样本协方差除以
治疗人数，加上对照列均值向量的样本协方差除以对照人数，再按层权重平方
合成。配对只流式枚举，不保存全部配对；分块大小和资料顺序不影响语义。
"""
from collections import defaultdict
from fractions import Fraction
import math

from .io import finite, identity, integer, positive, rational, unique

PRIORITY = ('survival', 'progression', 'quality_change')
LAYERS = PRIORITY
OUTCOMES = ('win', 'loss', 'tie', 'incomparable')


def _time_pair(treatment, control):
    """返回 (结果, 原因)；结果取 1/-1/0/None，均从治疗组视角。"""
    tt, et = treatment
    tc, ec = control
    if et and ec:
        if tt < tc:
            return -1, 'treatment_earlier_event'
        if tt > tc:
            return 1, 'control_earlier_event'
        return 0, 'events_same_time'
    if not et and not ec:
        return None, 'both_censored'
    if et:
        if tt < tc:
            return -1, 'treatment_earlier_event'
        if tt == tc:
            return None, 'event_censor_same_time'
        return None, 'censor_before_event'
    if tc < tt:
        return 1, 'control_earlier_event'
    if tc == tt:
        return None, 'event_censor_same_time'
    return None, 'censor_before_event'


def _quality_pair(tq, cq, tolerance):
    """返回 (结果, 原因)；差值严格大于容差才分胜负。"""
    if tq is None or cq is None:
        return None, 'quality_missing'
    difference = tq - cq
    if difference > tolerance:
        return 1, 'treatment_quality_better'
    if difference < -tolerance:
        return -1, 'control_quality_better'
    return 0, 'quality_within_tolerance'


def _compare_pair(treatment, control, tolerance):
    """按生存、无进展、生活质量依次比较，返回 (结果, 终止层, 原因)。"""
    for name in ('survival', 'progression'):
        value, reason = _time_pair(treatment[name], control[name])
        if value is None:
            return 'incomparable', name, reason
        if value != 0:
            return ('win' if value > 0 else 'loss'), name, reason
    value, reason = _quality_pair(treatment['quality_change'],
                                  control['quality_change'], tolerance)
    if value is None:
        return 'incomparable', 'quality_change', reason
    if value == 0:
        return 'tie', 'quality_change', reason
    return ('win' if value > 0 else 'loss'), 'quality_change', reason


def _time_value(row, name):
    outcome = row[name]
    if not isinstance(outcome, dict):
        raise ValueError('时间结局需要对象')
    time = finite(outcome['time'])
    if time < 0:
        raise ValueError('时间不能为负')
    if type(outcome['event']) is not bool:
        raise ValueError('事件标记需要布尔值')
    return (Fraction(time), outcome['event'])


def _subject(row):
    if not isinstance(row, dict):
        raise ValueError('受试者需要对象')
    subject_id = identity(row['id'])
    stratum = identity(row['stratum'])
    arm = row['arm']
    if arm not in ('treatment', 'control'):
        raise ValueError('未知组别')
    survival = _time_value(row, 'survival')
    progression = _time_value(row, 'progression')
    quality = row['quality_change']
    if quality is not None:
        quality = Fraction(finite(quality))
    return {
        'id': subject_id,
        'stratum': stratum,
        'arm': arm,
        'survival': survival,
        'progression': progression,
        'quality_change': quality,
    }


def _validate(value, block_size):
    if not isinstance(value, dict):
        raise ValueError('顶层需要对象')
    subjects = value['subjects']
    if not isinstance(subjects, list) or not subjects:
        raise ValueError('受试者列表不能为空')
    unique(subjects, 'id')
    rows = [_subject(row) for row in subjects]

    weights_in = value['stratum_weights']
    if not isinstance(weights_in, dict) or not weights_in:
        raise ValueError('层权重需要非空对象')
    weights = {}
    for stratum, weight in weights_in.items():
        identity(stratum)
        weights[stratum] = rational(weight)
        if weights[stratum] <= 0:
            raise ValueError('层权重需要正数')
    if sum(weights.values(), Fraction(0)) != 1:
        raise ValueError('层权重和必须为1')

    tolerance = Fraction(positive(value['quality_tolerance']))
    z = positive(value['normal_z'])

    if block_size is None:
        block_size = integer(value['pair_block_size'], 1)
    else:
        block_size = integer(block_size, 1)

    strata = {row['stratum'] for row in rows}
    if strata != set(weights):
        raise ValueError('层权重与受试者分层不一致')

    groups = defaultdict(lambda: {'treatment': [], 'control': []})
    for row in rows:
        groups[row['stratum']][row['arm']].append(row)
    for stratum, arms in groups.items():
        if not arms['treatment'] or not arms['control']:
            raise ValueError('每层都需要治疗组和对照组受试者')
        for arm in arms:
            arms[arm].sort(key=lambda row: row['id'])
    return rows, groups, weights, tolerance, z, block_size


def _pairs(treatments, controls, block_size):
    """按治疗行优先顺序流式产出跨组配对，每次只保留一个分块。"""
    total = len(treatments) * len(controls)
    for start in range(0, total, block_size):
        stop = min(start + block_size, total)
        for index in range(start, stop):
            yield treatments[index // len(controls)], controls[index % len(controls)]


def _projection_covariance(vectors, count):
    """向量的样本协方差除以人数：sum((v-mean)(v-mean)')/(n(n-1))。"""
    mean_w = sum((vector[0] for vector in vectors), Fraction(0)) / count
    mean_l = sum((vector[1] for vector in vectors), Fraction(0)) / count
    sww = sll = swl = Fraction(0)
    for win, loss in vectors:
        dw = win - mean_w
        dl = loss - mean_l
        sww += dw * dw
        swl += dw * dl
        sll += dl * dl
    denominator = count * (count - 1)
    return sww / denominator, swl / denominator, sll / denominator


def _exact(value):
    return str(Fraction(value))


def analyze(value, block_size=None, pair_sink=None):
    """分析一份已解析的研究资料。

    pair_sink 为可选的可调用对象，每个跨组配对调用一次，用来流式导出逐对
    明细；函数本身不保留配对列表。返回的字典不含来源 SHA-256。
    """
    rows, groups, weights, tolerance, z, block_size = _validate(value, block_size)

    stratum_results = []
    stratum_covariances = {}
    stratum_u = {}
    overall_counts = {outcome: 0 for outcome in OUTCOMES}
    overall_termination = {layer: {outcome: 0 for outcome in OUTCOMES} for layer in LAYERS}
    projections = {'treatment': [], 'control': []}
    unestimable = []

    for stratum in sorted(groups):
        treatments = groups[stratum]['treatment']
        controls = groups[stratum]['control']
        nt, nc = len(treatments), len(controls)
        row_wins = [0] * nt
        row_losses = [0] * nt
        column_wins = [0] * nc
        column_losses = [0] * nc
        counts = {outcome: 0 for outcome in OUTCOMES}
        termination = {layer: {outcome: 0 for outcome in OUTCOMES} for layer in LAYERS}
        index = {row['id']: i for i, row in enumerate(treatments)}
        control_index = {row['id']: j for j, row in enumerate(controls)}

        for treatment, control in _pairs(treatments, controls, block_size):
            result, layer, reason = _compare_pair(treatment, control, tolerance)
            counts[result] += 1
            termination[layer][result] += 1
            i = index[treatment['id']]
            j = control_index[control['id']]
            if result == 'win':
                row_wins[i] += 1
                column_wins[j] += 1
            elif result == 'loss':
                row_losses[i] += 1
                column_losses[j] += 1
            if pair_sink is not None:
                pair_sink({
                    'stratum': stratum,
                    'treatment_id': treatment['id'],
                    'control_id': control['id'],
                    'result': result,
                    'termination_layer': layer,
                    'reason': reason,
                })
        pairs = nt * nc
        u_win = Fraction(counts['win'], pairs)
        u_loss = Fraction(counts['loss'], pairs)
        u_tie = Fraction(counts['tie'], pairs)
        u_incomparable = Fraction(counts['incomparable'], pairs)
        stratum_u[stratum] = (u_win, u_loss)

        for i, treatment in enumerate(treatments):
            projections['treatment'].append({
                'stratum': stratum,
                'arm': 'treatment',
                'id': treatment['id'],
                'n_opponents': nc,
                'win_count': row_wins[i],
                'loss_count': row_losses[i],
                'win_mean': float(Fraction(row_wins[i], nc)),
                'loss_mean': float(Fraction(row_losses[i], nc)),
                'win_mean_exact': _exact(Fraction(row_wins[i], nc)),
                'loss_mean_exact': _exact(Fraction(row_losses[i], nc)),
            })
        for j, control in enumerate(controls):
            projections['control'].append({
                'stratum': stratum,
                'arm': 'control',
                'id': control['id'],
                'n_opponents': nt,
                'win_count': column_wins[j],
                'loss_count': column_losses[j],
                'win_mean': float(Fraction(column_wins[j], nt)),
                'loss_mean': float(Fraction(column_losses[j], nt)),
                'win_mean_exact': _exact(Fraction(column_wins[j], nt)),
                'loss_mean_exact': _exact(Fraction(column_losses[j], nt)),
            })

        estimable = nt >= 2 and nc >= 2
        components = None
        covariance = None
        if estimable:
            treatment_vectors = [
                (Fraction(row_wins[i], nc), Fraction(row_losses[i], nc))
                for i in range(nt)
            ]
            control_vectors = [
                (Fraction(column_wins[j], nt), Fraction(column_losses[j], nt))
                for j in range(nc)
            ]
            c_t = _projection_covariance(treatment_vectors, nt)
            c_c = _projection_covariance(control_vectors, nc)
            covariance = tuple(c_t[k] + c_c[k] for k in range(3))
            components = {'treatment': c_t, 'control': c_c}
            stratum_covariances[stratum] = covariance
        else:
            unestimable.append(stratum)

        for outcome in OUTCOMES:
            overall_counts[outcome] += counts[outcome]
        for layer in LAYERS:
            for outcome in OUTCOMES:
                overall_termination[layer][outcome] += termination[layer][outcome]

        stratum_results.append({
            'stratum': stratum,
            'weight': _exact(weights[stratum]),
            'weight_value': float(weights[stratum]),
            'n_treatment': nt,
            'n_control': nc,
            'pairs': pairs,
            'counts': dict(counts),
            'u_win': float(u_win),
            'u_loss': float(u_loss),
            'u_tie': float(u_tie),
            'u_incomparable': float(u_incomparable),
            'u_win_exact': _exact(u_win),
            'u_loss_exact': _exact(u_loss),
            'termination': {layer: dict(termination[layer]) for layer in LAYERS},
            'variance_estimable': estimable,
            'covariance': None if not estimable else {
                'w': float(covariance[0]),
                'l': float(covariance[2]),
                'wl': float(covariance[1]),
            },
            'covariance_exact': None if not estimable else {
                'w': _exact(covariance[0]),
                'l': _exact(covariance[2]),
                'wl': _exact(covariance[1]),
            },
            'covariance_components': None if not estimable else {
                'treatment': {
                    'w': _exact(components['treatment'][0]),
                    'l': _exact(components['treatment'][2]),
                    'wl': _exact(components['treatment'][1]),
                },
                'control': {
                    'w': _exact(components['control'][0]),
                    'l': _exact(components['control'][2]),
                    'wl': _exact(components['control'][1]),
                },
            },
        })

    w = sum((weights[result['stratum']] * stratum_u[result['stratum']][0]
             for result in stratum_results), Fraction(0))
    l = sum((weights[result['stratum']] * stratum_u[result['stratum']][1]
             for result in stratum_results), Fraction(0))

    variance_estimable = not unestimable
    covariance = None
    if variance_estimable:
        vw = vl = vwl = Fraction(0)
        for result in stratum_results:
            weight = weights[result['stratum']]
            cw, cwl, cl = stratum_covariances[result['stratum']]
            vw += weight * weight * cw
            vl += weight * weight * cl
            vwl += weight * weight * cwl
        covariance = (vw, vwl, vl)

    if w > 0 and l > 0:
        win_ratio = w / l
    else:
        win_ratio = None

    if w == 0 and l == 0:
        status = 'boundary_both_zero'
    elif l == 0:
        status = 'boundary_loss_zero'
    elif w == 0:
        status = 'boundary_win_zero'
    elif not variance_estimable:
        status = 'point_estimate_only'
    else:
        status = 'ok'

    se_log = None
    confidence_interval = None
    if status == 'ok':
        vw, vwl, vl = covariance
        variance_log = (vw / (w * w)) + (vl / (l * l)) - (2 * vwl / (w * l))
        if variance_log < 0:
            # 精确分数下确实为负：小样本投影协方差可给出非正方差，不强行取零。
            status = 'variance_not_positive'
        else:
            se_log = math.sqrt(float(variance_log))
            if not math.isfinite(se_log):
                se_log = None
                status = 'variance_not_positive'
            else:
                log_ratio = math.log(float(w / l))
                lower = math.exp(log_ratio - z * se_log)
                upper = math.exp(log_ratio + z * se_log)
                if math.isfinite(lower) and math.isfinite(upper):
                    confidence_interval = [lower, upper]
                else:
                    se_log = None
                    status = 'variance_not_positive'

    overall = {
        'w': float(w),
        'l': float(l),
        'w_exact': _exact(w),
        'l_exact': _exact(l),
        'win_ratio': None if win_ratio is None else float(win_ratio),
        'win_ratio_exact': None if win_ratio is None else _exact(win_ratio),
        'status': status,
        'variance_estimable': variance_estimable,
        'unestimable_strata': unestimable,
        'covariance': None if covariance is None else {
            'w': float(covariance[0]),
            'l': float(covariance[2]),
            'wl': float(covariance[1]),
        },
        'covariance_exact': None if covariance is None else {
            'w': _exact(covariance[0]),
            'l': _exact(covariance[2]),
            'wl': _exact(covariance[1]),
        },
        'log_win_ratio_se': se_log,
        'confidence_interval': confidence_interval,
        'counts': dict(overall_counts),
        'termination': overall_termination,
    }

    return {
        'protocol': {
            'priority': list(PRIORITY),
            'stratum_weights': {s: _exact(weight) for s, weight in weights.items()},
            'quality_tolerance': float(tolerance),
            'normal_z': z,
            'pair_block_size': block_size,
            'projection': 'two-sample U-statistic first-order projection; sample '
                          'covariance of row or column win/loss means divided by '
                          'arm count; weighted by squared stratum weights',
            'projection_orientation': 'win and loss counted from the treatment perspective',
            'scope': '本研究的预先指定配对规则，不外推为通用临床规范',
        },
        'overall': overall,
        'strata': stratum_results,
        'projections': {
            'axes': 'win 与 loss 在两组投影中都按治疗组视角计数，'
                    '因此行均值和列均值共享同一 W/L 坐标轴',
            'treatment': projections['treatment'],
            'control': projections['control'],
        },
    }
