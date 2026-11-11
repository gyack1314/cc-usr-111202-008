"""清点各层两组及时间结局缺失情况。"""
from collections import Counter
from .io import unique,finite,identity

def summarize(v):
    unique(v['subjects'],'id');counts=Counter();missing=0
    for r in v['subjects']:
        layer=identity(r['stratum'])
        if r['arm'] not in ('treatment','control'):raise ValueError('未知组别')
        counts[layer,r['arm']]+=1
        for name in ('survival','progression'):
            e=r[name]
            if finite(e['time'])<0 or type(e['event']) is not bool:raise ValueError('时间结局错误')
        if r['quality_change'] is None:missing+=1
        else:finite(r['quality_change'])
    return {'counts':[{'stratum':s,'arm':a,'n':n} for (s,a),n in sorted(counts.items())],'missing_quality':missing}
