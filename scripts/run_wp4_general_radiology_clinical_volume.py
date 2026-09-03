#!/usr/bin/env python3
"""Corrected WP4 disease-based TriNetX radiology utilization extraction.

Primary estimator: annual disease patient-year imaging utilization.
Sensitivity: imaging within +/-31 days of a qualifying diagnosis.
Only aggregate outputs are written. No patient-level artifacts are written.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import pandas as pd

YEARS=(2018,2019)
AGE_LABELS=("0-14 years","15-49 years","50-74 years","75+ years")
DX={
 "BC":{"ICD-9-CM":("174",),"ICD-10-CM":("C50",)},
 "COPD":{"ICD-9-CM":("490","491","492","493","494","495","496"),"ICD-10-CM":("J44",)},
 "CKD":{"ICD-9-CM":("585",),"ICD-10-CM":("N18",)},
 "CRC":{"ICD-9-CM":("153","154"),"ICD-10-CM":("C18","C19","C20")},
 "IHD":{"ICD-9-CM":("410","411","412","413","414"),"ICD-10-CM":("I20","I21","I22","I23","I24","I25")},
}
CPT={
 "BC":{"CT":("0633T","0634T","0635T","0636T","0637T","0638T"),"MRI":("77046","77047","77048","77049"),"US":("76641","76642")},
 "COPD":{"CT":("71250","71260","71270"),"X-ray":("71045","71046","71047","71048")},
 "CKD":{"CT":("74150","74160","74170","74174","74175","74176","74177","74178"),"MRI":("74181","74182","74183"),"US":("51798","76700")},
 "CRC":{"CT":("72192","72193","72194","74150","74160","74170","74176","74177","74178","74261","74262"),"MRI":("72195","72196","72197","74181","74182","74183"),"US":("76700",)},
 "IHD":{"CT":("71275","75571","75572","75573","75574"),"MRI":("71555",)},
}
STATE={
"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","DC":"District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}

def prog(i,n,msg):
 p=os.environ.get("RUNRELAY_PROGRESS_FILE")
 if not p:return
 q=Path(p);q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_suffix(q.suffix+".tmp")
 tmp.write_text(json.dumps({"schema_version":1,"current":i,"total":n,"fraction":i/n,"phase":msg,"updated_at_epoch":time.time()}));os.replace(tmp,q)

def date8(s):return s.astype("string").str.replace(r"\.0$","",regex=True).str.extract(r"(\d{8})",expand=False)
def norm(s):return s.astype("string").str.strip().str.upper().str.replace(".","",regex=False)
def csys(s):
 s=s.astype("string").str.upper();o=pd.Series(pd.NA,index=s.index,dtype="string");o[s.str.contains("ICD-10",na=False)]="ICD-10-CM";o[s.str.contains("ICD-9",na=False)]="ICD-9-CM";return o

def age_group(a):
 a=pd.to_numeric(a,errors="coerce");o=pd.Series(pd.NA,index=a.index,dtype="string");o[(a>=0)&(a<=14)]=AGE_LABELS[0];o[(a>=15)&(a<=49)]=AGE_LABELS[1];o[(a>=50)&(a<=74)]=AGE_LABELS[2];o[a>=75]=AGE_LABELS[3];return o

def table(root,names,required):
 for n in names:
  p=root/n
  if p.exists():return p
 for p in root.glob("*.csv"):
  try:c=set(pd.read_csv(p,nrows=0).columns)
  except Exception:continue
  if set(required)<=c:return p
 raise FileNotFoundError(f"table not found: {names}")

def read_dx(path,chunksize):
 out=[];cols=["patient_id","code_system","code","principal_diagnosis_indicator","date"]
 for x in pd.read_csv(path,dtype="string",usecols=cols,chunksize=chunksize,low_memory=False):
  x["date8"]=date8(x.date);x["year"]=pd.to_numeric(x.date8.str[:4],errors="coerce")
  x=x[x.year.isin(YEARS)&x.principal_diagnosis_indicator.str.upper().eq("P")].copy()
  if x.empty:continue
  x["code"]=norm(x.code);x["system"]=csys(x.code_system)
  for d,systems in DX.items():
   m=pd.Series(False,index=x.index)
   for s,prefs in systems.items():m|=x.system.eq(s)&x.code.str.startswith(tuple(prefs),na=False)
   if m.any():
    z=x.loc[m,["patient_id","year","date8"]].copy();z["disease"]=d;out.append(z)
 if not out:raise RuntimeError("No qualifying target diagnoses in 2018-2019")
 x=pd.concat(out,ignore_index=True).drop_duplicates();x["year"]=x.year.astype(int);return x

def zipmap(path):
 z=pd.read_csv(path,dtype="string",usecols=["zip","state"]);z["zip3"]=z.zip.str.zfill(5).str[:3];z.state=z.state.str.upper();z=z[z.state.isin(STATE)]
 n=z.groupby("zip3").state.nunique();amb=n[n>1].rename("n_states").reset_index();good=set(n[n==1].index)
 m=z[z.zip3.isin(good)][["zip3","state"]].drop_duplicates("zip3");m["state_name"]=m.state.map(STATE);return m,amb

def strata(cohort,patient_file,zip_file):
 ids=set(cohort.patient_id.astype(str));p=pd.read_csv(patient_file,dtype="string",usecols=["patient_id","sex","year_of_birth","postal_code"]);p=p[p.patient_id.isin(ids)].drop_duplicates("patient_id")
 zm,amb=zipmap(zip_file);p["zip3"]=p.postal_code.str.replace(r"\.0$","",regex=True).str.zfill(3).str[:3];p=p.merge(zm,on="zip3",how="left")
 x=cohort.merge(p,on="patient_id",how="left");x["sex"]=x.sex.str.upper();x["age"]=x.year-pd.to_numeric(x.year_of_birth,errors="coerce");x["age_group"]=age_group(x.age);x["gbd_compatible"]=x.sex.isin(["M","F"])&x.age_group.notna()&x.state_name.notna()
 qc=x.assign(missing_state=x.state_name.isna(),missing_age=x.age_group.isna(),invalid_sex=~x.sex.isin(["M","F"])).groupby(["disease","year"],as_index=False).agg(n_patient_years=("patient_id","size"),n_gbd_compatible=("gbd_compatible","sum"),n_missing_state=("missing_state","sum"),n_missing_age=("missing_age","sum"),n_invalid_sex=("invalid_sex","sum"))
 return x,amb,qc

def cpt_frame():
 rows=[]
 for d,mods in CPT.items():
  for m,codes in mods.items():
   for c in codes:rows.append((d,m,c))
 return pd.DataFrame(rows,columns=["disease","modality","CPTcode"])

def read_px(path,cohort,chunksize):
 cmap=cpt_frame();lookup={}
 for r in cmap.itertuples(index=False): lookup.setdefault(r.CPTcode,[]).append((r.disease,r.modality))
 ids=set(cohort.patient_id.astype(str));dset={d:set(zip(g.patient_id.astype(str),g.year.astype(int))) for d,g in cohort.groupby("disease")};out=[]
 for x in pd.read_csv(path,dtype="string",usecols=["patient_id","code","date"],chunksize=chunksize,low_memory=False):
  x=x[x.patient_id.isin(ids)].copy();x["code"]=norm(x.code);x=x[x.code.isin(lookup)]
  if x.empty:continue
  x["date8"]=date8(x.date);x["year"]=pd.to_numeric(x.date8.str[:4],errors="coerce");x=x[x.year.isin(YEARS)]
  rows=[]
  for r in x[["patient_id","code","date8","year"]].itertuples(index=False):
   for d,m in lookup.get(r.code,[]):
    if (str(r.patient_id),int(r.year)) in dset[d]:rows.append((r.patient_id,d,int(r.year),r.date8,m))
  if rows:out.append(pd.DataFrame(rows,columns=["patient_id","disease","year","date8","modality"]))
 if not out:return pd.DataFrame(columns=["patient_id","disease","year","date8","modality"])
 return pd.concat(out,ignore_index=True).drop_duplicates(["patient_id","disease","year","date8","modality"])

def window31(events,dx,days=31):
 if events.empty:return events.copy()
 e=events.copy();d=dx.copy();e["ed"]=pd.to_datetime(e.date8,format="%Y%m%d",errors="coerce");d["dd"]=pd.to_datetime(d.date8,format="%Y%m%d",errors="coerce")
 m=e.merge(d[["patient_id","disease","year","dd"]],on=["patient_id","disease","year"],how="left");m=m[(m.ed-m.dd).abs()<=pd.Timedelta(days=days)]
 return m[["patient_id","disease","year","date8","modality"]].drop_duplicates()

def aggregate(cohort,events,window):
 c=cohort[cohort.gbd_compatible].copy();keys=["disease","year","state_name","sex","age_group"]
 den=c.groupby(keys,dropna=False).patient_id.nunique().rename("n_patients").reset_index()
 if events.empty:num=pd.DataFrame(columns=keys+["modality","n_procedures"])
 else:num=events.merge(c[["patient_id","disease","year","state_name","sex","age_group"]],on=["patient_id","disease","year"],how="inner").groupby(keys+["modality"],dropna=False).size().rename("n_procedures").reset_index()
 mods=cpt_frame()[["disease","modality"]].drop_duplicates();grid=den.merge(mods,on="disease",how="inner");z=grid.merge(num,on=keys+["modality"],how="left");z.n_procedures=z.n_procedures.fillna(0).astype(int);z["procedures_per_patient"]=z.n_procedures/z.n_patients;z["utilization_window"]=window;return z.rename(columns={"state_name":"state"})

def args():
 p=argparse.ArgumentParser();p.add_argument("--trinetx-dir",type=Path,required=True);p.add_argument("--zip-map",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--chunksize",type=int,default=750000);return p.parse_args()

def main():
 a=args();o=a.output_dir;o.mkdir(parents=True,exist_ok=True);prog(0,7,"resolve input tables")
 dxfile=table(a.trinetx_dir,["diagnosis.csv"],["patient_id","code_system","code","principal_diagnosis_indicator","date"]);pxfile=table(a.trinetx_dir,["procedure.csv","procedures.csv"],["patient_id","code","date"]);pfile=table(a.trinetx_dir,["patient.csv"],["patient_id","sex","year_of_birth","postal_code"])
 prog(1,7,"extract target principal diagnoses");dxe=read_dx(dxfile,a.chunksize);cohort=dxe[["patient_id","disease","year"]].drop_duplicates()
 prog(2,7,"attach GBD-compatible strata");cs,amb,qc=strata(cohort,pfile,a.zip_map);amb.to_csv(o/"ambiguous_zip3_prefixes.csv",index=False);qc.to_csv(o/"cohort_qc.csv",index=False)
 prog(3,7,"extract disease-relevant imaging");ev=read_px(pxfile,cohort,a.chunksize)
 prog(4,7,"aggregate annual utilization");annual=aggregate(cs,ev,"annual");annual.to_csv(o/"trinetx_imaging_utilization_annual_long.csv",index=False)
 prog(5,7,"aggregate +/-31-day source sensitivity");d31=aggregate(cs,window31(ev,dxe),"diagnosis31d");d31.to_csv(o/"trinetx_imaging_utilization_diagnosis31d_long.csv",index=False)
 cpt_frame().to_csv(o/"cpt_disease_modality_map.csv",index=False);pd.DataFrame([(d,s,p) for d,v in DX.items() for s,ps in v.items() for p in ps],columns=["disease","code_system","prefix"]).to_csv(o/"disease_diagnosis_prefixes.csv",index=False)
 nat=pd.concat([annual,d31]).groupby(["utilization_window","disease","year","modality"],as_index=False).agg(n_patients=("n_patients","sum"),n_procedures=("n_procedures","sum"));nat["procedures_per_patient"]=nat.n_procedures/nat.n_patients;nat.to_csv(o/"trinetx_imaging_utilization_national_window_comparison.csv",index=False)
 overlap=cohort.groupby("patient_id").disease.nunique().value_counts().sort_index();meta={"status":"WP4_GENERAL_RADIOLOGY_CLINICAL_VOLUME_OK","years":list(YEARS),"primary_window":"annual","sensitivity_window":"diagnosis31d","zero_imaging_patient_years_in_denominator":True,"multi_target_disease_patients_excluded":False,"same_day_deduplication":"patient+disease+year+date+modality","cross_stratum_fallback":False,"crc_icd10":"C18+C19+C20","copd_icd9_status":"provisional source definition 490-496; adjudicate before freeze","patient_level_files_written":False,"n_disease_patient_years":int(len(cohort)),"n_unique_patients":int(cohort.patient_id.nunique()),"target_disease_count_per_patient":{str(int(k)):int(v) for k,v in overlap.items()},"inputs":{"diagnosis":str(dxfile),"procedure":str(pxfile),"patient":str(pfile),"zip_map":str(a.zip_map)}};(o/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
 prog(7,7,"complete");print(json.dumps(meta,indent=2));print(f"Wrote aggregate outputs to {o}")
if __name__=="__main__":main()
