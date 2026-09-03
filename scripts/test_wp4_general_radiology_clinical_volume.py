#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/"scripts"/"run_wp4_general_radiology_clinical_volume.py";SCALER=ROOT/"scripts"/"scale_wp4_general_radiology_gbd.py"
def w(p,t):p.write_text(t.strip()+"\n")
def main():
 with tempfile.TemporaryDirectory() as td0:
  td=Path(td0);tri=td/"tri";out=td/"out";tri.mkdir();out.mkdir()
  w(tri/"diagnosis.csv","""patient_id,code_system,code,principal_diagnosis_indicator,date
p1,ICD-10-CM,C50.911,P,20180115
p2,ICD-10-CM,N18.3,P,20180210
p2,ICD-10-CM,I25.10,P,20180305
p3,ICD-10-CM,C20,P,20190710
p4,ICD-10-CM,J44.9,P,20190401
p5,ICD-10-CM,C50.919,S,20180501
p6,ICD-10-CM,C18.9,P,20181201""")
  w(tri/"patient.csv","""patient_id,sex,year_of_birth,postal_code
p1,F,1970,170
p2,M,1940,100
p3,F,1980,303
p4,M,1960,900
p5,F,1975,170
p6,M,2004,170""")
  w(tri/"procedure.csv","""patient_id,code,date
p1,77046,20180120
p1,77047,20180120
p1,76641,20181201
p2,74150,20180220
p2,75571,20180310
p3,74261,20190820
p3,72195,20191220
p6,74261,20181215""")
  z=td/"zip.csv";w(z,"""zip,state
17033,PA
10001,NY
30301,GA
90001,CA""")
  subprocess.run([sys.executable,str(SCRIPT),"--trinetx-dir",str(tri),"--zip-map",str(z),"--output-dir",str(out),"--chunksize","3"],check=True)
  a=pd.read_csv(out/"trinetx_imaging_utilization_annual_long.csv");d=pd.read_csv(out/"trinetx_imaging_utilization_diagnosis31d_long.csv")
  assert ((a.disease=="CRC")&(a.year==2019)&(a.state=="Georgia")).any()
  assert ((a.disease=="CRC")&(a.year==2018)&(a.age_group=="0-14 years")).any()
  r=a[(a.disease=="BC")&(a.modality=="MRI")].iloc[0];assert r.n_procedures==1
  assert a[(a.disease=="BC")&(a.modality=="US")].iloc[0].n_procedures==1
  assert d[(d.disease=="BC")&(d.modality=="US")].iloc[0].n_procedures==0
  r=a[(a.disease=="COPD")&(a.modality=="CT")].iloc[0];assert r.n_patients==1 and r.n_procedures==0
  assert ((a.disease=="CKD")&(a.year==2018)).any() and ((a.disease=="IHD")&(a.year==2018)).any()
  assert '"patient_level_files_written": false' in (out/"run_metadata.json").read_text()
  g=td/"gbd.csv";w(g,"""measure_id,metric_id,location_id,location_name,sex_name,age_name,cause_name,year,val,upper,lower
5,1,533,Georgia,Female,15-49 years,Colon and rectum cancer,2019,400,440,360
5,1,35,Georgia,Female,15-49 years,Colon and rectum cancer,2019,9999,9999,9999""");scaled=td/"scaled"
  subprocess.run([sys.executable,str(SCALER),"--utilization-dir",str(out),"--gbd-file",str(g),"--output-dir",str(scaled)],check=True)
  sx=pd.read_csv(scaled/"gbd_scaled_imaging_by_stratum.csv");assert 35 not in set(sx.location_id);assert sx[(sx.disease=="CRC")&(sx.modality=="CT")&(sx.utilization_window=="annual")].iloc[0].gbd_n==400
  print("WP4 synthetic tests passed.")
if __name__=="__main__":main()
