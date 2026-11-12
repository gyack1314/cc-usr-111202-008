"""核对按优先级比较的胜比、投影协方差及流式分块语义。"""
from collections import Counter
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

from research_data.cli import main as cli_main
from research_data.io import load
from research_data.winratio import (_compare_pair, _projection_covariance,
                                    _pairs, _time_pair, analyze)

TOLERANCE = Fraction(1)


def _subject(subject_id, arm, survival, progression, quality, stratum='s'):
    return {
        'id': subject_id,
        'stratum': stratum,
        'arm': arm,
        'survival': survival,
        'progression': progression,
        'quality_change': quality,
    }


def _material(subjects, weights=None, tolerance=1.0, z=1.95996398454):
    return {
        'subjects': subjects,
        'stratum_weights': weights or {'s': '1'},
        'quality_tolerance': tolerance,
        'normal_z': z,
        'pair_block_size': 1,
    }


def _pair(treatment, control, tolerance=TOLERANCE):
    return _compare_pair(treatment, control, tolerance)


class TimeOutcomeTests(unittest.TestCase):
    def test_earlier_observed_event_is_loss(self):
        self.assertEqual(_time_pair((Fraction(5), True), (Fraction(9), True)),
                         (-1, 'treatment_earlier_event'))
        self.assertEqual(_time_pair((Fraction(9), True), (Fraction(5), True)),
                         (1, 'control_earlier_event'))

    def test_same_time_events_are_tie(self):
        self.assertEqual(_time_pair((Fraction(5), True), (Fraction(5), True)),
                         (0, 'events_same_time'))

    def test_event_and_censor_at_same_time_unknown(self):
        self.assertEqual(_time_pair((Fraction(5), True), (Fraction(5), False)),
                         (None, 'event_censor_same_time'))
        self.assertEqual(_time_pair((Fraction(5), False), (Fraction(5), True)),
                         (None, 'event_censor_same_time'))

    def test_censoring_relative_to_observed_event(self):
        # 对照删失在治疗事件之后：治疗在观察期内已有事件，判为劣势。
        self.assertEqual(_time_pair((Fraction(4), True), (Fraction(9), False)),
                         (-1, 'treatment_earlier_event'))
        # 治疗删失发生较早：对照事件时点未知，整对不可比较。
        self.assertEqual(_time_pair((Fraction(4), False), (Fraction(9), True)),
                         (None, 'censor_before_event'))
        # 对照的已观察事件早于治疗删失：对照劣势，治疗获胜。
        self.assertEqual(_time_pair((Fraction(9), False), (Fraction(4), True)),
                         (1, 'control_earlier_event'))
        self.assertEqual(_time_pair((Fraction(9), False), (Fraction(4), False)),
                         (None, 'both_censored'))


class PriorityTests(unittest.TestCase):
    def _row(self, survival, progression, quality):
        return {'survival': survival, 'progression': progression,
                'quality_change': quality}

    def test_survival_decides_before_progression(self):
        treatment = self._row((Fraction(5), True), (Fraction(1), True), 100)
        control = self._row((Fraction(9), True), (Fraction(2), True), 0)
        self.assertEqual(_pair(treatment, control),
                         ('loss', 'survival', 'treatment_earlier_event'))

    def test_progression_decides_when_survival_ties(self):
        treatment = self._row((Fraction(5), True), (Fraction(9), True), 100)
        control = self._row((Fraction(5), True), (Fraction(11), True), 0)
        self.assertEqual(_pair(treatment, control),
                         ('loss', 'progression', 'treatment_earlier_event'))

    def test_quality_decides_when_both_times_tie(self):
        treatment = self._row((Fraction(5), True), (Fraction(9), True), 5)
        control = self._row((Fraction(5), True), (Fraction(9), True), 1)
        self.assertEqual(_pair(treatment, control),
                         ('win', 'quality_change', 'treatment_quality_better'))

    def test_unknown_time_stops_without_falling_through(self):
        treatment = self._row((Fraction(5), False), (Fraction(1), True), 100)
        control = self._row((Fraction(9), True), (Fraction(2), True), 0)
        self.assertEqual(_pair(treatment, control),
                         ('incomparable', 'survival', 'censor_before_event'))
        treatment = self._row((Fraction(5), True), (Fraction(1), False), 100)
        control = self._row((Fraction(5), True), (Fraction(9), True), 0)
        self.assertEqual(_pair(treatment, control),
                         ('incomparable', 'progression', 'censor_before_event'))

    def test_quality_tolerance_and_missing(self):
        treatment = self._row((Fraction(5), True), (Fraction(9), True), 2)
        control = self._row((Fraction(5), True), (Fraction(9), True), 1)
        self.assertEqual(_pair(treatment, control),
                         ('tie', 'quality_change', 'quality_within_tolerance'))
        treatment = self._row((Fraction(5), True), (Fraction(9), True), 3)
        control = self._row((Fraction(5), True), (Fraction(9), True), 1)
        self.assertEqual(_pair(treatment, control),
                         ('win', 'quality_change', 'treatment_quality_better'))
        treatment = self._row((Fraction(5), True), (Fraction(9), True), None)
        control = self._row((Fraction(5), True), (Fraction(9), True), 0)
        self.assertEqual(_pair(treatment, control),
                         ('incomparable', 'quality_change', 'quality_missing'))


class CovarianceTests(unittest.TestCase):
    def test_projection_covariance_matches_formula(self):
        vectors = [(Fraction(1, 3), Fraction(2, 3)),
                   (Fraction(1, 3), Fraction(0)),
                   (Fraction(3, 3), Fraction(0))]
        cww, cwl, cll = _projection_covariance(vectors, 3)
        mean_w = Fraction(5, 9)
        mean_l = Fraction(2, 9)
        expected_ww = sum(((w - mean_w) ** 2 for w, _ in vectors), Fraction(0)) / 6
        expected_wl = sum(((w - mean_w) * (l - mean_l) for w, l in vectors),
                          Fraction(0)) / 6
        expected_ll = sum(((l - mean_l) ** 2 for _, l in vectors), Fraction(0)) / 6
        self.assertEqual(cww, expected_ww)
        self.assertEqual(cwl, expected_wl)
        self.assertEqual(cll, expected_ll)
        self.assertEqual((cww, cwl, cll), (Fraction(4, 81), Fraction(-2, 81),
                                           Fraction(4, 81)))

    def test_single_person_arm_has_no_variance(self):
        # 治疗组只有 1 人，点估计 1/2 可列出，但两组都要至少 2 人才估方差。
        subjects = [
            _subject('t0', 'treatment', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 1),
            _subject('c0', 'control', {'time': 12, 'event': True},
                     {'time': 12, 'event': True}, 0),
            _subject('c1', 'control', {'time': 5, 'event': True},
                     {'time': 5, 'event': True}, 0),
        ]
        result = analyze(_material(subjects))
        self.assertEqual(result['overall']['status'], 'point_estimate_only')
        self.assertFalse(result['overall']['variance_estimable'])
        self.assertIsNone(result['overall']['log_win_ratio_se'])
        self.assertEqual(result['overall']['win_ratio_exact'], '1')
        self.assertEqual(result['strata'][0]['variance_estimable'], False)
        self.assertIsNone(result['strata'][0]['covariance'])

    def test_boundary_zero_loss_keeps_state(self):
        # 治疗组两个人都比对照晚出现已观察事件，因此全部获胜；L=0。
        subjects = [
            _subject('t0', 'treatment', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 1),
            _subject('t1', 'treatment', {'time': 11, 'event': True},
                     {'time': 11, 'event': True}, 1),
            _subject('c0', 'control', {'time': 4, 'event': True},
                     {'time': 4, 'event': True}, 0),
            _subject('c1', 'control', {'time': 6, 'event': True},
                     {'time': 6, 'event': True}, 0),
        ]
        result = analyze(_material(subjects))
        overall = result['overall']
        self.assertEqual(overall['status'], 'boundary_loss_zero')
        self.assertIsNone(overall['win_ratio'])
        self.assertIsNone(overall['win_ratio_exact'])
        self.assertEqual(overall['l_exact'], '0')
        self.assertEqual(overall['w_exact'], '1')
        # 没有偷偷加 0.5：L 保持为零，不给出有限区间。
        self.assertIsNone(overall['log_win_ratio_se'])
        self.assertIsNone(overall['confidence_interval'])

    def test_pairs_not_treated_as_independent(self):
        # 两个治疗者各自横扫两个对照：W=L=1/2，四个配对彼此高度相关。
        subjects = [
            _subject('t0', 'treatment', {'time': 5, 'event': True},
                     {'time': 5, 'event': True}, 1),
            _subject('t1', 'treatment', {'time': 5, 'event': True},
                     {'time': 5, 'event': True}, 1),
            _subject('c0', 'control', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 0),
            _subject('c1', 'control', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 0),
        ]
        # 让 t0 全胜、t1 全负，W 与 L 都为正。
        subjects[1]['survival'] = {'time': 12, 'event': True}
        subjects[2]['survival'] = {'time': 6, 'event': True}
        subjects[3]['survival'] = {'time': 6, 'event': True}
        result = analyze(_material(subjects))
        overall = result['overall']
        self.assertEqual(overall['status'], 'ok')
        self.assertEqual(overall['w_exact'], '1/2')
        self.assertEqual(overall['l_exact'], '1/2')
        self.assertEqual(overall['covariance_exact'],
                         {'w': '1/4', 'l': '1/4', 'wl': '-1/4'})
        # 把四个配对当独立伯努利会得到 SE=1；投影方差得到 SE=2。
        self.assertEqual(overall['log_win_ratio_se'], 2.0)


class StreamingAndOrderTests(unittest.TestCase):
    def _material(self, subjects, block_size=1):
        material = _material(subjects)
        material['pair_block_size'] = block_size
        return material

    def _sample(self, stratum='s'):
        return [
            _subject('t0', 'treatment', {'time': 5, 'event': True},
                     {'time': 5, 'event': True}, 4, stratum),
            _subject('t1', 'treatment', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, None, stratum),
            _subject('t2', 'treatment', {'time': 7, 'event': False},
                     {'time': 7, 'event': False}, 1, stratum),
            _subject('c0', 'control', {'time': 6, 'event': True},
                     {'time': 6, 'event': True}, 2, stratum),
            _subject('c1', 'control', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 0, stratum),
            _subject('c2', 'control', {'time': 8, 'event': False},
                     {'time': 8, 'event': False}, 3, stratum),
        ]

    def test_pair_generator_covers_all_cross_pairs_once(self):
        treatments = [{'id': f't{i}'} for i in range(3)]
        controls = [{'id': f'c{j}'} for j in range(4)]
        pairs = list(_pairs(treatments, controls, 2))
        self.assertEqual(len(pairs), 12)
        self.assertEqual(Counter((t['id'], c['id']) for t, c in pairs),
                         Counter((f't{i}', f'c{j}')
                                 for i in range(3) for j in range(4)))

    def test_block_size_does_not_change_semantics(self):
        material = self._material(self._sample())
        baseline = analyze(material, block_size=1)
        for size in (2, 3, 5, 100):
            with self.subTest(block_size=size):
                result = analyze(material, block_size=size)
                # 分块大小只是执行参数，会如实记录，但数值结果不变。
                self.assertEqual(result['protocol']['pair_block_size'], size)
                self.assertEqual(result['overall'], baseline['overall'])
                self.assertEqual(result['strata'], baseline['strata'])
                self.assertEqual(result['projections'], baseline['projections'])

    def test_record_order_does_not_change_semantics(self):
        material = self._material(self._sample())
        baseline = analyze(material, block_size=2)
        material['subjects'].reverse()
        reordered = analyze(material, block_size=3)
        self.assertEqual(reordered['overall'], baseline['overall'])
        self.assertEqual(reordered['strata'], baseline['strata'])
        self.assertEqual(reordered['projections'], baseline['projections'])

    def test_pairs_stream_without_being_retained(self):
        material = self._material(self._sample())
        seen = []
        result = analyze(material, block_size=2, pair_sink=seen.append)
        self.assertEqual(len(seen), 9)
        self.assertEqual(sum(1 for pair in seen if pair['result'] == 'win')
                         + sum(1 for pair in seen if pair['result'] == 'loss')
                         + sum(1 for pair in seen if pair['result'] == 'tie')
                         + sum(1 for pair in seen if pair['result'] == 'incomparable'),
                         9)
        self.assertEqual(result['overall']['counts'], {
            outcome: sum(1 for pair in seen if pair['result'] == outcome)
            for outcome in ('win', 'loss', 'tie', 'incomparable')})
        self.assertEqual(sorted(pair['stratum'] for pair in seen), ['s'] * 9)

    def test_projection_counts_recompute_covariance(self):
        material = self._material(self._sample())
        result = analyze(material, block_size=3)
        self.assertEqual(result['overall']['status'], 'ok')
        stratum = result['strata'][0]
        self.assertEqual((stratum['pairs'],
                          stratum['counts']['win'] + stratum['counts']['loss']
                          + stratum['counts']['tie']
                          + stratum['counts']['incomparable']), (9, 9))
        treatment = [row for row in result['projections']['treatment']]
        control = [row for row in result['projections']['control']]
        self.assertEqual(sum(row['win_count'] for row in treatment),
                         stratum['counts']['win'])
        self.assertEqual(sum(row['loss_count'] for row in control),
                         stratum['counts']['loss'])
        self.assertEqual(sum(row['win_count'] for row in control),
                         stratum['counts']['win'])
        # 逐人投影重算：用人数减1归一的人均值协方差相加。
        for arm_rows, count, key in ((treatment, 3, 'treatment'),
                                     (control, 3, 'control')):
            mean_w = sum(Fraction(row['win_count'], row['n_opponents'])
                         for row in arm_rows) / count
            mean_l = sum(Fraction(row['loss_count'], row['n_opponents'])
                         for row in arm_rows) / count
            ww = sum(((Fraction(row['win_count'], row['n_opponents']) - mean_w) ** 2
                      for row in arm_rows), Fraction(0)) / (count * (count - 1))
            ll = sum(((Fraction(row['loss_count'], row['n_opponents']) - mean_l) ** 2
                      for row in arm_rows), Fraction(0)) / (count * (count - 1))
            wl = sum(((Fraction(row['win_count'], row['n_opponents']) - mean_w)
                      * (Fraction(row['loss_count'], row['n_opponents']) - mean_l)
                      for row in arm_rows), Fraction(0)) / (count * (count - 1))
            self.assertEqual(stratum['covariance_components'][key]['w'], str(ww))
            self.assertEqual(stratum['covariance_components'][key]['l'], str(ll))
            self.assertEqual(stratum['covariance_components'][key]['wl'], str(wl))

    def test_termination_layers_report_early_stop(self):
        material = self._material(self._sample())
        result = analyze(material, block_size=2)
        stratum = result['strata'][0]
        total = sum(outcome_count
                    for layer in stratum['termination'].values()
                    for outcome_count in layer.values())
        self.assertEqual(total, stratum['pairs'])
        self.assertEqual(sum(stratum['termination']['survival'].values()), 8)
        survival = stratum['termination']['survival']
        self.assertEqual(survival['incomparable'], 3)
        self.assertEqual(survival['win'], 2)
        self.assertEqual(survival['loss'], 3)
        # 一对因生存层两个删失未知而不可比较；另一对生存同刻事件为确定平局，
        # 才下移到进展层，但进展也未观察，仍未继续。
        self.assertEqual(stratum['termination']['progression'],
                         {'win': 0, 'loss': 0, 'tie': 0, 'incomparable': 0})
        # 一对生存与进展都是确定平局，生活质量因缺失而不可比较。
        self.assertEqual(stratum['termination']['quality_change'],
                         {'win': 0, 'loss': 0, 'tie': 0, 'incomparable': 1})


class VarianceEstimationTests(unittest.TestCase):
    def test_mixed_strata_make_overall_variance_unestimable(self):
        subjects = [
            # a 层可估方差：治疗对 c0 获胜、对 c1 失败。
            _subject('a-t0', 'treatment', {'time': 9, 'event': True},
                     {'time': 9, 'event': True}, 1, 'a'),
            _subject('a-t1', 'treatment', {'time': 11, 'event': True},
                     {'time': 11, 'event': True}, 1, 'a'),
            _subject('a-c0', 'control', {'time': 4, 'event': True},
                     {'time': 4, 'event': True}, 0, 'a'),
            _subject('a-c1', 'control', {'time': 13, 'event': True},
                     {'time': 13, 'event': True}, 0, 'a'),
            # b 层治疗组只有 1 人：点估计可列出，方差不可估。
            _subject('b-t0', 'treatment', {'time': 5, 'event': True},
                     {'time': 5, 'event': True}, 1, 'b'),
            _subject('b-c0', 'control', {'time': 7, 'event': True},
                     {'time': 7, 'event': True}, 0, 'b'),
        ]
        material = _material(subjects, weights={'a': '1/2', 'b': '1/2'})
        result = analyze(material)
        overall = result['overall']
        self.assertEqual(overall['status'], 'point_estimate_only')
        self.assertEqual(overall['unestimable_strata'], ['b'])
        self.assertIsNone(overall['covariance'])
        self.assertIsNone(overall['log_win_ratio_se'])
        self.assertEqual(overall['w_exact'], '1/4')
        self.assertEqual(overall['l_exact'], '3/4')
        self.assertEqual(overall['win_ratio_exact'], '1/3')

    def test_rejects_mismatched_weights(self):
        subjects = self._sample_material()['subjects']
        # 权重和不为 1、出现资料中没有的层、缺少有受试者的层都需要拒绝。
        for weights in ({'s': '1/2'}, {'s': '2/3', 'x': '1/3'},
                        {'s': '1/3', 'x': '1/3'}):
            with self.subTest(weights=weights):
                material = _material(subjects, weights=weights)
                with self.assertRaises(ValueError):
                    analyze(material)

    def _sample_material(self):
        return _material(StreamingAndOrderTests()._sample())

    def test_approximate_interval_uses_given_z(self):
        material = self._sample_material()
        result = analyze(material, block_size=2)
        overall = result['overall']
        self.assertEqual(overall['status'], 'ok')
        se = overall['log_win_ratio_se']
        ratio = float(Fraction(overall['win_ratio_exact']))
        lower, upper = overall['confidence_interval']
        z = 1.95996398454
        self.assertAlmostEqual(lower, math.exp(math.log(ratio) - z * se))
        self.assertAlmostEqual(upper, math.exp(math.log(ratio) + z * se))


class MaterialAnalysisTests(unittest.TestCase):
    def test_primary_material_analysis(self):
        value, digest = load('data/normal.json')
        result = analyze(value)
        self.assertEqual(len(digest), 64)
        overall = result['overall']
        self.assertEqual(overall['status'], 'ok')
        self.assertEqual(overall['counts'],
                         {'win': 6, 'loss': 4, 'tie': 0, 'incomparable': 8})
        # 生存层决定了全部配对：删失既不是胜也不是负，整对不可比较。
        self.assertEqual(overall['termination']['survival']['incomparable'], 8)
        self.assertEqual(overall['w_exact'], '1/3')
        self.assertEqual(overall['l_exact'], '2/9')
        self.assertEqual(overall['win_ratio_exact'], '3/2')
        self.assertEqual(overall['covariance_exact'],
                         {'w': '13/225', 'l': '13/405', 'wl': '-13/675'})
        self.assertEqual(overall['log_win_ratio_se'], 1.3)
        self.assertEqual(result['protocol']['stratum_weights'],
                         {'one': '2/5', 'two': '3/5'})

    def test_all_survival_unknown_keeps_boundary_state(self):
        value, _ = load('data/all-survival-unknown.json')
        result = analyze(value)
        overall = result['overall']
        self.assertEqual(overall['status'], 'boundary_both_zero')
        self.assertIsNone(overall['win_ratio'])
        self.assertIsNone(overall['confidence_interval'])
        self.assertEqual(overall['counts']['incomparable'], 18)
        # 生存层未知时不下移到进展或生活质量。
        self.assertEqual(overall['termination']['progression'],
                         {'win': 0, 'loss': 0, 'tie': 0, 'incomparable': 0})
        self.assertEqual(overall['termination']['quality_change'],
                         {'win': 0, 'loss': 0, 'tie': 0, 'incomparable': 0})

    def test_one_person_arm_lists_point_estimate_only(self):
        value, _ = load('data/one-person-arm.json')
        result = analyze(value)
        overall = result['overall']
        self.assertEqual(overall['status'], 'point_estimate_only')
        self.assertEqual(overall['unestimable_strata'], ['one'])
        self.assertEqual(overall['win_ratio_exact'], '5/6')
        self.assertIsNone(overall['covariance'])
        self.assertIsNone(overall['confidence_interval'])
        self.assertEqual(result['strata'][0]['variance_estimable'], False)
        self.assertEqual(result['strata'][1]['variance_estimable'], True)

    def test_invalid_material_is_rejected(self):
        value, _ = load('data/invalid-domain.json')
        with self.assertRaises(ValueError):
            analyze(value)


class CliTests(unittest.TestCase):
    def test_analyze_exports_summary_and_streamed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.json'
            pairs = Path(directory) / 'pairs.jsonl'
            argv = ['research_data', 'analyze', 'data/normal.json',
                    '--pairs', str(pairs), '--output', str(summary),
                    '--block-size', '2']
            with unittest.mock.patch.object(sys, 'argv', argv):
                cli_main()
            exported = json.loads(summary.read_text(encoding='utf8'))
            self.assertEqual(len(exported['source_sha256']), 64)
            self.assertEqual(exported['analysis']['protocol']['pair_block_size'], 2)
            lines = pairs.read_text(encoding='utf8').splitlines()
            self.assertEqual(len(lines), exported['analysis']['overall']['counts'][
                'win'] + exported['analysis']['overall']['counts']['loss']
                + exported['analysis']['overall']['counts']['tie']
                + exported['analysis']['overall']['counts']['incomparable'])
            for line in lines:
                pair = json.loads(line)
                self.assertEqual(set(pair), {'stratum', 'treatment_id',
                                             'control_id', 'result',
                                             'termination_layer', 'reason'})
            # 不传 --output 时输出到标准输出，仍带来源指纹和同一分析。
            argv = ['research_data', 'analyze', 'data/normal.json']
            buffer = io.StringIO()
            with unittest.mock.patch.object(sys, 'argv', argv), \
                    unittest.mock.patch.object(sys, 'stdout', buffer):
                cli_main()
            stdout_value = json.loads(buffer.getvalue())
            self.assertEqual(stdout_value['analysis']['overall'],
                             exported['analysis']['overall'])
