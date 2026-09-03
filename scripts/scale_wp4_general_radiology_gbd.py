#!/usr/bin/env python3
"""Scale corrected WP4 TriNetX utilization to GBD 2021 US-state prevalence.

No pooled fallback is used. Missing TriNetX state/sex/age/year cells remain
missing and are reported. US states are selected by numeric GBD location ID to
avoid mixing the country Georgia with the US state Georgia.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
YEARS=(2018,2019);AGES=("0-14 years","15-49 years","50-74 years","75+ years")
CAUSE={"Breast cancer":"BC","Chronic obstructive pulmonary disease":"COPD","Chronic kidney disease":"CKD","Colon and rectum cancer":"CRC","Ischemic heart disease":"IHD"}
STATE_IDS={"Alabama":523,"Alaska":524,"Arizona":525,"Arkansas":526,"California":527,"Colorado":528,"Connecticut":529,"Delaware":530,"District of Columbia":531,"Florida":532,"Georgia":533,"Hawaii":534,"Idaho":535,"Illinois":536,"Indiana":537,"Iowa":538,"Kansas":539,"Kentucky":540,"Louisiana":541,"Maine":542,"Maryland":543,"Massachusetts":544,"Michigan":545,"Minnesota":546,"Mississippi":547,"Missouri":548,"Montana":549,"Nebraska":550,"Nevada":551,"New Hampshire":552,"New Jersey":553,"New Mexico":554,"New York":555,"North Carolina":556,"North Dakota":557,"Ohio":558,"Oklahoma":559,"Oregon":560,"Pennsylvania":561,"Rhode Island":562,"South Carolina":563,"South Dakota":564,"Tennessee":565,"Texas":566,"Utah":567,"Vermont":568,"Virginia":569,"Washington":570,"West Virginia":571,"Wisconsin":572,"Wyoming":573}
def load_gbd(p):
 g=pd.read_csv(p);need={"measure_id","metric_id","location_id","location_name","sex_name","age_name","cause_name","year","val","upper","lower"};miss=need-set(g.columns)
 if miss:raise ValueError(f"GBD file missing columns: {sorted(miss)}")
 g=g[(g.measure_id==5)&(g.metric_id==1)&g.year.isin(YEARS)&g.age_name.isin(AGES)&g.cause_name.isin(CAUSE)&g.location_id.isin(set(STATE_IDS.values()))].copy();g["disease"]=g.cause_name.map(CAUSE);g["sex"]=g.sex_name.map({"Male":"M","Female":"F"});g=g[g.sex.isin(["M","F"])]
 expected=g.location_name.map(STATE_IDS);bad=g[expected.ne(g.location_id)]
 if len(bad):raise ValueError("GBD state ID/name mismatch; refusing name-only geographic join")
 key=["disease","year","location_name","sex","age_name"]
 if g.duplicated(key).any():raise ValueError("Duplicate GBD US-state prevalence strata")
 return g[key+["location_id","val","lower","upper"]].rename(columns={"location_name":"state","age_name":"age_group","val":"gbd_n"})
def main():
 p=argparse.ArgumentParser();p.add_argument("--utilization-dir",type=Path,required=True);p.add_argument("--gbd-file",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 files=[a.utilization_dir/"trinetx_imaging_utilization_annual_long.csv",a.utilization_dir/"trinetx_imaging_utilization_diagnosis31d_long.csv"];u=pd.concat([pd.read_csv(f) for f in files],ignore_index=True);g=load_gbd(a.gbd_file);g.to_csv(a.output_dir/"gbd_us_state_prevalence_2018_2019.csv",index=False)
 key=["disease","year","state","sex","age_group"];mods=u[["disease","modality"]].drop_duplicates();s=g.merge(mods,on="disease").merge(u,on=key+["modality"],how="left",indicator=True);s["trinetx_stratum_available"]=s._merge.eq("both");s=s.drop(columns="_merge");s["scaled_n_procedures"]=s.procedures_per_patient*s.gbd_n;s["scaled_n_procedures_lower"]=s.procedures_per_patient*s.lower;s["scaled_n_procedures_upper"]=s.procedures_per_patient*s.upper;s.to_csv(a.output_dir/"gbd_scaled_imaging_by_stratum.csv",index=False);s[~s.trinetx_stratum_available].to_csv(a.output_dir/"gbd_strata_missing_trinetx_rates.csv",index=False)
 n=s.groupby(["utilization_window","disease","year","modality"],as_index=False).agg(gbd_n=("gbd_n","sum"),scaled_n_procedures=("scaled_n_procedures",lambda x:x.sum(min_count=1)),scaled_n_procedures_lower=("scaled_n_procedures_lower",lambda x:x.sum(min_count=1)),scaled_n_procedures_upper=("scaled_n_procedures_upper",lambda x:x.sum(min_count=1)),n_gbd_strata=("gbd_n","size"),n_trinetx_strata_available=("trinetx_stratum_available","sum"));n["strata_coverage"]=n.n_trinetx_strata_available/n.n_gbd_strata;n.to_csv(a.output_dir/"gbd_scaled_imaging_national.csv",index=False)
 meta={"status":"WP4_GBD_SCALING_OK","gbd_measure":"Prevalence","gbd_metric":"Number","years":list(YEARS),"cross_stratum_fallback":False,"us_state_selection":"numeric GBD location IDs 523-573 including DC","country_georgia_excluded":True,"n_gbd_rows":int(len(g)),"n_scaled_rows":int(len(s))};(a.output_dir/"gbd_scaling_metadata.json").write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
