import io, os, requests, pandas as pd, numpy as np

DEFAULT_URL='https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2026/2026-07-07/ultimate_ufc_dataset.csv'

def download_dataset(url=DEFAULT_URL,path='data/fights.csv'):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    r=requests.get(url,timeout=90); r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content)); df.to_csv(path,index=False); return df

def load(path='data/fights.csv'): return pd.read_csv(path)

def clean(df):
    d=df.copy()
    date_col=next((c for c in ['date','Date','EVENT_DATE','event_date'] if c in d.columns),None)
    d['_date']=pd.to_datetime(d[date_col],errors='coerce') if date_col else pd.NaT
    if 'r_fighter' in d.columns and 'b_fighter' in d.columns:
        d['_A']=d['r_fighter'].astype(str).str.strip(); d['_B']=d['b_fighter'].astype(str).str.strip()
    else:
        pairs=[('fighter_1','fighter_2'),('fighter1','fighter2'),('R_fighter','B_fighter'),('red_fighter','blue_fighter')]
        for a,b in pairs:
            if a in d.columns and b in d.columns:
                d['_A']=d[a].astype(str).str.strip(); d['_B']=d[b].astype(str).str.strip(); break
    if '_A' not in d: raise ValueError('Could not identify fighter-name columns.')
    if 'winner' in d.columns:
        w=d['winner'].astype(str).str.strip().str.lower()
        d['_winner']=np.where(w.eq('red'),d['_A'],np.where(w.eq('blue'),d['_B'],np.where(w.eq('draw'),'',w)))
    elif 'Winner' in d.columns:
        w=d['Winner'].astype(str).str.strip().str.lower()
        d['_winner']=np.where(w.eq('red'),d['_A'],np.where(w.eq('blue'),d['_B'],np.where(w.eq('draw'),'',w)))
    elif 'outcome' in d.columns:
        w=d['outcome'].astype(str).str.lower()
        d['_winner']=np.where(w.isin(['fighter','fighter1','red']),d['_A'],np.where(w.isin(['fighter2','blue']),d['_B'],''))
    else: raise ValueError('Could not identify winner/outcome columns.')
    d['_y']=np.where(d['_winner'].str.lower()==d['_A'].str.lower(),1,np.where(d['_winner'].str.lower()==d['_B'].str.lower(),0,np.nan))
    d=d.dropna(subset=['_y']).copy(); d['_y']=d['_y'].astype(int)
    if d['_date'].notna().sum()>0: d=d.sort_values('_date').reset_index(drop=True)
    if d['_y'].nunique()<2: raise ValueError('Training data contains fewer than two outcome classes. Refresh data and try again.')
    return d

def find_col(d,side,keys):
    prefixes=(['R_','red_','r_'] if side=='A' else ['B_','blue_','b_'])
    cols=list(d.index) if isinstance(d,pd.Series) else list(d.columns)
    for k in keys:
        for p in prefixes:
            for candidate in (p+k,p+k.lower(),p+k.upper()):
                if candidate in cols:return candidate
    return None

def num(v):
    try:return float(str(v).replace('%','').strip()) if not pd.isna(v) else 0.0
    except:return 0.0

def fight_observation(row,side):
    mapping={'kd':['kd'],'sig_str':['avg_sig_str_landed','sig_str','SIG_STR'],'td':['avg_td_landed','td','TD'],'sub':['avg_sub_att','sub_att','SUB_ATT']}
    out={}
    for k,keys in mapping.items():
        c=find_col(row,side,keys); out[k]=num(row[c]) if c else 0.0
    return out
