import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(
    page_title="RailAI | Railway Operations Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp{background:#f5f7fb;color:#172033}
.block-container{max-width:1240px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"]{display:none}
.hero{background:linear-gradient(135deg,#102a56,#174d91);color:white;padding:30px 34px;border-radius:18px;margin-bottom:25px}
.hero h1{color:white!important;margin:0 0 6px}
.hero p{color:#dce9f8;margin:0}
.card{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px 26px 20px;margin:22px 0}
.step{color:#1764b5;font-size:.8rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase}
.result{background:#f7faff;border:1px solid #cfe0f5;border-radius:14px;padding:18px;margin-top:18px}
.ok{background:#eef8f2;border:1px solid #cce8d6;border-radius:11px;padding:13px 15px;color:#236a40}
.warn{background:#fff8e9;border:1px solid #efdfb6;border-radius:11px;padding:13px 15px;color:#735814}
.stButton>button{background:#145ca8;color:white;border:0;border-radius:9px;font-weight:650}
.stButton>button:hover{background:#0f4b8d;color:white}
div[data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:12px 16px}
</style>
""", unsafe_allow_html=True)

ROOT=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(ROOT,"models","priority_model.pkl")

if "priority" not in st.session_state: st.session_state.priority=None
if "priority_source" not in st.session_state: st.session_state.priority_source=""
if "mpriority" not in st.session_state: st.session_state.mpriority=50
if "block_result" not in st.session_state: st.session_state.block_result=None
if "schedule" not in st.session_state: st.session_state.schedule=None
if "emergency_result" not in st.session_state: st.session_state.emergency_result=None

@st.cache_resource
def load_model():
    if joblib and os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except Exception: return None
    return None
MODEL=load_model()

DEFAULT={"region":"Central Railway","season":"Monsoon","train_type":"Express","train_age_years":8,
"average_speed_kmph":90,"distance_travelled_km":650,"track_temperature_c":38,"rail_wear_mm":3.4,
"track_vibration_level":.62,"ballast_condition":"Fair","track_curvature_degree":1.8,
"ambient_temperature_c":31,"humidity_percent":72,"rainfall_mm":18,"wind_speed_kmph":16,
"wheel_wear_percent":28,"axle_temperature_c":67,"brake_pressure_psi":92,"brake_pad_wear_percent":31,
"bearing_temperature_c":71,"battery_voltage":110,"traction_motor_temp_c":84,
"signal_system_status":"Healthy","power_consumption_kw":520,"load_factor_percent":78,
"daily_trips":8,"delay_minutes":12,"last_maintenance_days":110,"inspection_score":76,
"sensor_health_index":91}
HIGH={**DEFAULT,"rail_wear_mm":5.1,"track_vibration_level":.88,"wheel_wear_percent":55,
"axle_temperature_c":94,"bearing_temperature_c":99,"traction_motor_temp_c":112,
"delay_minutes":38,"last_maintenance_days":280,"load_factor_percent":92,
"inspection_score":51,"sensor_health_index":72}
LOW={**DEFAULT,"rail_wear_mm":1.4,"track_vibration_level":.22,"wheel_wear_percent":10,
"axle_temperature_c":55,"bearing_temperature_c":58,"traction_motor_temp_c":63,
"delay_minutes":3,"last_maintenance_days":35,"load_factor_percent":52,
"inspection_score":93,"sensor_health_index":98}

def fallback(v):
    def n(x,a,b): return np.clip((x-a)/(b-a),0,1)
    s=100*(.18*n(v["rail_wear_mm"],0,6)+.12*n(v["track_vibration_level"],0,1)+
    .10*n(v["wheel_wear_percent"],0,80)+.08*n(v["axle_temperature_c"],40,110)+
    .08*n(v["bearing_temperature_c"],40,110)+.06*n(v["traction_motor_temp_c"],40,130)+
    .10*n(v["delay_minutes"],0,60)+.10*n(v["last_maintenance_days"],0,365)+
    .06*n(v["load_factor_percent"],0,100)+.04*n(v["daily_trips"],0,15)+
    .05*(1-n(v["inspection_score"],0,100))+.03*(1-n(v["sensor_health_index"],0,100)))
    return round(float(np.clip(s,0,100)),2)

def get_score(v):
    if MODEL is not None:
        try:
            return round(float(np.clip(MODEL.predict(pd.DataFrame([v]))[0],0,100)),2),"Random Forest priority model"
        except Exception: pass
    return fallback(v),"Prototype fallback priority engine"

def level(s):
    return "Critical" if s>=80 else "High" if s>=60 else "Medium" if s>=40 else "Normal"

def hhmm(m):
    m=int(m)%1440
    return f"{m//60:02d}:{m%60:02d}"

TRAINS={
"Section-A":[("T001",60),("T002",180),("T003",300),("T004",420),("T005",540),("T006",660),("T007",840),("T008",1020),("T009",1200),("T010",1320)],
"Section-B":[("T011",120),("T012",240),("T013",360),("T014",480),("T015",600),("T016",720),("T017",900),("T018",1080),("T019",1260),("T020",1380)]}

def candidates(section,duration,preferred,traffic,night):
    out=[]; penalty={"Low":2,"Medium":5,"High":10}[traffic]
    for s in range(0,1440-duration+1,30):
        e=s+duration
        if not night and (s<360 or e>1320): continue
        trains=[tid for tid,t in TRAINS[section] if s<=t<e]
        dist=min(abs(s-preferred),1440-abs(s-preferred))
        opt=100-dist/14-penalty*len(trains)
        out.append({"start":s,"end":e,"affected":len(trains),"trains":", ".join(trains) if trains else "None","score":round(opt,2)})
    return sorted(out,key=lambda x:x["score"],reverse=True)

def base_tasks():
    return [
    {"task_id":"MT-001","department":"Engineering","maintenance_type":"Track Repair","location_km":45.2,"priority":44.75,"section":"Section-A","start":90,"duration":90,"status":"UNCHANGED"},
    {"task_id":"MT-002","department":"S&T","maintenance_type":"Signal Repair","location_km":47.8,"priority":39.18,"section":"Section-A","start":0,"duration":60,"status":"UNCHANGED"},
    {"task_id":"MT-003","department":"Traction","maintenance_type":"OHE Maintenance","location_km":52.1,"priority":36.88,"section":"Section-B","start":0,"duration":120,"status":"UNCHANGED"},
    {"task_id":"MT-004","department":"Engineering","maintenance_type":"Track Inspection","location_km":55.4,"priority":31.90,"section":"Section-B","start":150,"duration":60,"status":"UNCHANGED"}]

def timeline(ts,title):
    fig=go.Figure()
    for i,t in enumerate(ts):
        s=t["start"]; e=s+t["duration"]
        fig.add_trace(go.Scatter(x=[s,e,e,s,s],y=[i,i,i+.7,i+.7,i],fill="toself",mode="lines",name=t["task_id"]))
    fig.update_layout(title=title,height=max(300,70*len(ts)),showlegend=False,
        margin=dict(l=10,r=15,t=45,b=35),
        xaxis=dict(title="Time",range=[0,1440],tickmode="array",
        tickvals=list(range(0,1441,120)),ticktext=[hhmm(x) for x in range(0,1441,120)]),
        yaxis=dict(showticklabels=False))
    return fig

st.markdown('<div class="hero"><h1>RailAI Operations Control</h1><p>AI-powered maintenance prioritization, optimal block planning and emergency schedule replanning for Indian Railways.</p></div>',unsafe_allow_html=True)

# STEP 1
st.markdown('<div class="card"><div class="step">Step 01</div>',unsafe_allow_html=True)
st.header("Maintenance Assessment")
st.write("Enter the current railway condition. The priority engine ranks the existing maintenance requirement; it does not decide whether maintenance is needed.")

preset=st.selectbox("Condition preset",["Balanced condition","High-risk condition","Low-risk condition"])
d={"Balanced condition":DEFAULT,"High-risk condition":HIGH,"Low-risk condition":LOW}[preset]

a,b,c=st.columns(3)
with a:
    region=st.selectbox("Railway zone / region",["Central Railway","Northern Railway","Eastern Railway","Western Railway","South Central Railway","Southern Railway"])
    season=st.selectbox("Season",["Summer","Monsoon","Winter","Spring"],index=1)
    train_type=st.selectbox("Train type",["Express","Passenger","Freight","Superfast"])
    train_age=st.number_input("Train age (years)",0.,40.,float(d["train_age_years"]),.5)
    speed=st.number_input("Average speed (km/h)",0.,180.,float(d["average_speed_kmph"]),1.)
    distance=st.number_input("Distance travelled (km)",0.,5000.,float(d["distance_travelled_km"]),10.)
    track_temp=st.number_input("Track temperature (°C)",-20.,80.,float(d["track_temperature_c"]),.5)
    rail_wear=st.number_input("Rail wear (mm)",0.,10.,float(d["rail_wear_mm"]),.1)
    vibration=st.number_input("Track vibration level",0.,1.,float(d["track_vibration_level"]),.01)
    ballast=st.selectbox("Ballast condition",["Good","Fair","Poor"],index=1)
with b:
    curvature=st.number_input("Track curvature (degree)",0.,10.,float(d["track_curvature_degree"]),.1)
    ambient=st.number_input("Ambient temperature (°C)",-20.,60.,float(d["ambient_temperature_c"]),.5)
    humidity=st.number_input("Humidity (%)",0.,100.,float(d["humidity_percent"]),1.)
    rainfall=st.number_input("Rainfall (mm)",0.,500.,float(d["rainfall_mm"]),1.)
    wind=st.number_input("Wind speed (km/h)",0.,150.,float(d["wind_speed_kmph"]),1.)
    wheel=st.number_input("Wheel wear (%)",0.,100.,float(d["wheel_wear_percent"]),1.)
    axle=st.number_input("Axle temperature (°C)",20.,150.,float(d["axle_temperature_c"]),1.)
    brake_pressure=st.number_input("Brake pressure (psi)",0.,150.,float(d["brake_pressure_psi"]),1.)
    brake_pad=st.number_input("Brake pad wear (%)",0.,100.,float(d["brake_pad_wear_percent"]),1.)
    bearing=st.number_input("Bearing temperature (°C)",20.,150.,float(d["bearing_temperature_c"]),1.)
with c:
    battery=st.number_input("Battery voltage",0.,200.,float(d["battery_voltage"]),1.)
    motor=st.number_input("Traction motor temperature (°C)",20.,180.,float(d["traction_motor_temp_c"]),1.)
    signal=st.selectbox("Signal system status",["Healthy","Degraded","Fault"])
    power=st.number_input("Power consumption (kW)",0.,2000.,float(d["power_consumption_kw"]),10.)
    load=st.number_input("Load factor (%)",0.,100.,float(d["load_factor_percent"]),1.)
    trips=st.number_input("Daily trips",0,30,int(d["daily_trips"]),1)
    delay=st.number_input("Delay (minutes)",0.,300.,float(d["delay_minutes"]),1.)
    last_maint=st.number_input("Days since last maintenance",0.,1000.,float(d["last_maintenance_days"]),1.)
    inspection=st.number_input("Inspection score",0.,100.,float(d["inspection_score"]),1.)
    sensor=st.number_input("Sensor health index",0.,100.,float(d["sensor_health_index"]),1.)

vals={"region":region,"season":season,"train_type":train_type,"train_age_years":train_age,"average_speed_kmph":speed,
"distance_travelled_km":distance,"track_temperature_c":track_temp,"rail_wear_mm":rail_wear,
"track_vibration_level":vibration,"ballast_condition":ballast,"track_curvature_degree":curvature,
"ambient_temperature_c":ambient,"humidity_percent":humidity,"rainfall_mm":rainfall,"wind_speed_kmph":wind,
"wheel_wear_percent":wheel,"axle_temperature_c":axle,"brake_pressure_psi":brake_pressure,
"brake_pad_wear_percent":brake_pad,"bearing_temperature_c":bearing,"battery_voltage":battery,
"traction_motor_temp_c":motor,"signal_system_status":signal,"power_consumption_kw":power,
"load_factor_percent":load,"daily_trips":trips,"delay_minutes":delay,"last_maintenance_days":last_maint,
"inspection_score":inspection,"sensor_health_index":sensor}

if st.button("Calculate Maintenance Priority",use_container_width=True,key="calculate_priority"):
    st.session_state.priority,st.session_state.priority_source=get_score(vals)
    st.session_state.mpriority=int(round(st.session_state.priority))

if st.session_state.priority is not None:
    s=st.session_state.priority
    st.markdown('<div class="result">',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.metric("Maintenance priority",f"{s:.2f}/100")
    y.metric("Priority level",level(s))
    z.metric("Engine","Random Forest" if "Random Forest" in st.session_state.priority_source else "Fallback")
    st.progress(s/100)
    st.caption(st.session_state.priority_source)
    st.markdown('</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)

# STEP 2
st.markdown('<div class="card"><div class="step">Step 02</div>',unsafe_allow_html=True)
st.header("Optimal Block Planning")
st.write("The planner searches feasible windows and balances maintenance priority, preferred time and train traffic.")
a,b,c=st.columns(3)
with a:
    mtype=st.selectbox("Maintenance type",["Track Repair","Signal Repair","OHE Maintenance","Track Inspection","Bridge Maintenance"],key="mtype")
    section=st.selectbox("Section",["Section-A","Section-B"],key="section")
    km=st.number_input("Location (KM)",0.,2000.,45.2,.1,key="km")
with b:
    duration=st.select_slider("Required duration (minutes)",options=[30,60,90,120,150,180],value=90,key="duration")
    ptime=st.time_input("Preferred start time",value=datetime.strptime("01:30","%H:%M").time(),key="ptime")
    traffic=st.selectbox("Traffic level",["Low","Medium","High"],index=1,key="traffic")
with c:
    pdefault=int(round(st.session_state.priority if st.session_state.priority is not None else 50))
    priority=st.slider("Maintenance priority",0,100,pdefault,key="mpriority")
    night=st.checkbox("Allow night blocks",value=True,key="night")
    st.caption("Priority is carried forward from Step 01 and can be adjusted.")

if st.button("Generate Optimal Block",use_container_width=True,key="block"):
    pref=ptime.hour*60+ptime.minute
    cs=candidates(section,duration,pref,traffic,night)
    r=cs[0]
    st.session_state.block_result={**r,"maintenance_type":mtype,"section":section,"location_km":km,"priority":priority,"duration":duration,"traffic":traffic}
    ts=base_tasks()
    ts.append({"task_id":"MT-NEW","department":"Authority","maintenance_type":mtype,"location_km":km,"priority":priority,"section":section,"start":r["start"],"duration":duration,"status":"NEW"})
    st.session_state.schedule=ts

if st.session_state.block_result:
    r=st.session_state.block_result
    st.markdown('<div class="result">',unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Recommended block",f"{hhmm(r['start'])}–{hhmm(r['end'])}")
    q2.metric("Priority",f"{r['priority']}/100")
    q3.metric("Affected trains",r["affected"])
    q4.metric("Optimization score",f"{r['score']:.1f}")
    st.markdown(f'<div class="ok">Recommended window: <b>{hhmm(r["start"])} to {hhmm(r["end"])}</b> in {r["section"]}. Affected trains: <b>{r["trains"]}</b>.</div>',unsafe_allow_html=True)
    pref=ptime.hour*60+ptime.minute
    tab=pd.DataFrame([{"Start":hhmm(x["start"]),"End":hhmm(x["end"]),"Affected trains":x["affected"],"Trains":x["trains"],"Optimization score":x["score"]} for x in candidates(r["section"],r["duration"],pref,r["traffic"],night)[:6]])
    st.markdown("**Top candidate windows**")
    st.dataframe(tab,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.plotly_chart(timeline(st.session_state.schedule,"24-hour maintenance block plan"),use_container_width=True)
st.markdown('</div>',unsafe_allow_html=True)

# STEP 3
st.markdown('<div class="card"><div class="step">Step 03</div>',unsafe_allow_html=True)
st.header("Emergency Replanning")
st.write("Introduce a critical defect. The replanner protects the emergency block first and moves lower-priority maintenance when necessary.")
a,b,c=st.columns(3)
with a:
    etype=st.selectbox("Emergency type",["Critical Track Defect","Signal Failure","OHE Failure","Bridge Issue"],key="etype")
    ekm=st.number_input("Emergency location (KM)",0.,2000.,45.,.1,key="ekm")
    esec=st.selectbox("Emergency section",["Section-A","Section-B"],key="esec")
with b:
    detection=st.time_input("Detection time",value=datetime.strptime("23:00","%H:%M").time(),key="detection")
    edur=st.select_slider("Required repair duration",options=[30,60,90,120,150,180],value=90,key="edur")
    eprio=st.slider("Emergency priority",50,100,98,key="eprio")
with c:
    dept=st.selectbox("Department",["Engineering","S&T","Traction","Bridge & Works"],key="dept")
    action=st.text_input("Required action",value="Immediate inspection and repair",key="action")

if st.button("Replan Schedule",use_container_width=True,key="replan"):
    old=st.session_state.schedule if st.session_state.schedule else base_tasks()
    detect=detection.hour*60+detection.minute
    working=[]
    for t in old:
        nt=dict(t)
        if nt["section"]==esec and nt["start"]<detect: nt["start"]+=1440
        working.append(nt)
    em={"task_id":"EMG-001","department":dept,"maintenance_type":etype,"location_km":ekm,"priority":eprio,
        "section":esec,"start":detect,"duration":edur,"status":"EMERGENCY-NEW"}
    occupied=[(detect,detect+edur)]
    for t in sorted([x for x in working if x["section"]==esec],key=lambda x:x["priority"],reverse=True):
        s=t["start"];e=s+t["duration"]
        if e<=detect or s>=detect+edur:
            occupied.append((s,e));continue
        ns=max(detect+edur,s)
        while any(ns<oe and ns+t["duration"]>os for os,oe in occupied): ns+=30
        if t["priority"]<eprio:
            t["start"]=ns;t["status"]="RESCHEDULED"
        occupied.append((t["start"],t["start"]+t["duration"]))
    after=sorted([em]+working,key=lambda x:(x["section"],x["start"]))
    rows=[{"Task":t["task_id"],"New block":f"{hhmm(t['start'])}–{hhmm(t['start']+t['duration'])}","Status":"RESCHEDULED","Reason":"Moved to protect higher-priority emergency maintenance"} for t in after if t["status"]=="RESCHEDULED"]
    st.session_state.emergency_result={"emergency":em,"before":old,"after":after,"rows":rows}
    st.session_state.schedule=after

if st.session_state.emergency_result:
    er=st.session_state.emergency_result;em=er["emergency"]
    st.markdown('<div class="result">',unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Emergency block",f"{hhmm(em['start'])}–{hhmm(em['start']+em['duration'])}")
    q2.metric("Emergency priority",f"{em['priority']}/100")
    q3.metric("Rescheduled tasks",len(er["rows"]))
    q4.metric("Section",em["section"])
    if er["rows"]:
        st.markdown('<div class="warn">Replanning was triggered because the emergency priority is higher than overlapping lower-priority maintenance. The affected work was moved to the next feasible slot.</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(er["rows"]),use_container_width=True,hide_index=True)
    else:
        st.markdown('<div class="ok">Emergency maintenance was inserted without moving a lower-priority maintenance task.</div>',unsafe_allow_html=True)
    st.markdown("**Before vs after schedule**")
    x,y=st.columns(2)
    with x: st.plotly_chart(timeline(er["before"],"Before emergency"),use_container_width=True)
    with y: st.plotly_chart(timeline(er["after"],"After emergency replanning"),use_container_width=True)
    final=pd.DataFrame([{"Task":t["task_id"],"Type":t["maintenance_type"],"Section":t["section"],"Priority":t["priority"],"Block":f"{hhmm(t['start'])}–{hhmm(t['start']+t['duration'])}","Status":t["status"]} for t in er["after"]])
    st.markdown("**Updated schedule**")
    st.dataframe(final,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#738096;font-size:.82rem;margin-top:28px">Prototype decision-support system. AI recommendations are advisory; final block authorization remains with designated railway operating and maintenance personnel.</div>',unsafe_allow_html=True)
