import io, os, requests, pandas as pd, numpy as np

DEFAULT_URL='https://raw.githubusercontent.com/larissapavan/ufc-historical-fight-dataset/main/fights.csv'

def download_dataset(url=DEFAULT_URL,path='data/fights.csv'):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    r=requests.get(url,timeout=60); r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content)); df.to_csv(path,index=False); return df

def load(path='data/fights.csv'): return pd.read_csv(path)

def clean(df):
    d=df.copy()
    date_col=next((c for c in ['event_date','date','Date','EVENT_DATE'] if c in d.columns),None)
    d['_date']=pd.to_datetime(d[date_col],errors='coerce') if date_col else pd.NaT
    pairs=[('fighter_1','fighter_2'),('fighter1','fighter2'),('R_fighter','B_fighter'),('red_fighter','blue_fighter'),('f1_name','f2_name'),('f1','f2')]
    for a,b in pairs:
        if a in d.columns and b in d.columns:
            d['_A']=d[a].astype(str).str.strip(); d['_B']=d[b].astype(str).str.strip(); break
    if '_A' not in d: raise ValueError('Could not identify fighter-name columns.')
    if 'winner' in d.columns:
        w=d['winner'].astype(str).str.strip(); d['_y']=np.where(w.str.lower()==d['_A'].str.lower(),1,np.where(w.str.lower()==d['_B'].str.lower(),0,np.nan))
    elif 'Winner' in d.columns:
        w=d['Winner'].astype(str).str.strip(); d['_y']=np.where(w.str.lower()==d['_A'].str.lower(),1,np.where(w.str.lower()==d['_B'].str.lower(),0,np.nan))
    elif 'outcome' in d.columns:
        o=d['outcome'].astype(str).str.lower(); d['_y']=np.where(o.isin(['fighter','fighter1','1','win','w']),1,np.where(o.isin(['fighter2','0','loss','l']),0,np.nan))
    elif 'result' in d.columns:
        o=d['result'].astype(str).str.lower(); d['_y']=np.where(o.isin(['fighter','fighter1','1','win','w']),1,np.where(o.isin(['fighter2','0','loss','l']),0,np.nan))
    elif 'f1_result' in d.columns and 'f2_result' in d.columns:
        d['_y']=np.where(d['f1_result'].astype(str).str.upper().eq('W'),1,np.where(d['f2_result'].astype(str).str.upper().eq('W'),0,np.nan))
    else:
        # Current larissapavan source orders the winner first.
        d['_y']=1.0
    d=d.dropna(subset=['_y']).copy(); d['_y']=d['_y'].astype(int)
    if d['_date'].notna().sum()>0: d=d.sort_values('_date').reset_index(drop=True)
    return d

def find_col(d,side,keys):
    cols=list(d.index) if isinstance(d,pd.Series) else list(d.columns)
    prefixes=['R_','red_','fighter1_','fighter_1_','f1_'] if side=='A' else ['B_','blue_','fighter2_','fighter_2_','f2_']
    for k in keys:
        for p in prefixes:
            for candidate in (p+k,p+k.lower(),p+k.upper()):
                if candidate in cols:return candidate
    return None

def num(v):
    try:return float(str(v).replace('%','').strip()) if not pd.isna(v) else 0.0
    except:return 0.0

def fight_observation(row,side):
    mapping={'sig_str':['sig_str','SIG_STR','sig_str_landed'],'td':['td','TD','takedown'],'sub':['sub_att','SUB_ATT','submission'],'kd':['kd','KD','knockdown']}
    out={}
    for k,keys in mapping.items():
        c=find_col(row,side,keys); out[k]=num(row[c]) if c else 0.0
    return out
