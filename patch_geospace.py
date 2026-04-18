
# Patch script: replaces the GeoSpace section in frontend/app.py
keep_up_to = 42381  # character index where old GeoSpace section starts

old_content = open('frontend/app.py', 'r', encoding='utf-8').read()
keep = old_content[:keep_up_to]

new_geo = r'''# ══════════════════════════════════════════════════════════════════════════════
# GEOSPACE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "GeoSpace":
    st.markdown("""
    <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;color:white;margin-bottom:0.3rem">🌍 Geo-Spatial Intelligence</h1>
    <p style="color:rgba(255,255,255,0.55);font-size:1rem;margin-bottom:1.5rem">
        Interactive coral reef research site atlas — All major reef zones across India
    </p>
    """, unsafe_allow_html=True)

    REEF_SITES = [
        # Andaman & Nicobar
        {"name":"Havelock Island (Radhanagar)","region":"Andaman Nicobar","lat":12.00,"lon":92.95,"health":"Healthy","species":387,"area_km2":18.4,"rfhi":8.2,"sst":28.1,"threat":"Low"},
        {"name":"Neil Island Reef","region":"Andaman Nicobar","lat":11.83,"lon":92.72,"health":"Healthy","species":312,"area_km2":9.1,"rfhi":7.8,"sst":28.3,"threat":"Low"},
        {"name":"Wandoor MNP (S.Andaman)","region":"Andaman Nicobar","lat":11.55,"lon":92.45,"health":"Bleached","species":228,"area_km2":281.5,"rfhi":3.4,"sst":30.8,"threat":"High"},
        {"name":"Cinque Island","region":"Andaman Nicobar","lat":11.47,"lon":92.68,"health":"Healthy","species":340,"area_km2":22.8,"rfhi":8.6,"sst":27.9,"threat":"Low"},
        {"name":"Ritchie's Archipelago","region":"Andaman Nicobar","lat":12.10,"lon":93.10,"health":"Healthy","species":410,"area_km2":45.0,"rfhi":8.9,"sst":27.7,"threat":"Low"},
        {"name":"Barren Island (Volcanic)","region":"Andaman Nicobar","lat":12.28,"lon":93.85,"health":"Bleached","species":104,"area_km2":7.0,"rfhi":2.8,"sst":31.2,"threat":"Critical"},
        {"name":"Little Andaman Reef","region":"Andaman Nicobar","lat":10.60,"lon":92.55,"health":"Healthy","species":295,"area_km2":31.0,"rfhi":7.5,"sst":28.5,"threat":"Low"},
        {"name":"North Bay Reef","region":"Andaman Nicobar","lat":11.68,"lon":92.72,"health":"Bleached","species":187,"area_km2":5.2,"rfhi":3.1,"sst":30.4,"threat":"High"},
        {"name":"Car Nicobar Reef","region":"Andaman Nicobar","lat":9.15,"lon":92.82,"health":"Healthy","species":265,"area_km2":19.0,"rfhi":7.2,"sst":28.7,"threat":"Medium"},
        {"name":"Great Nicobar Reef","region":"Andaman Nicobar","lat":7.00,"lon":93.82,"health":"Healthy","species":278,"area_km2":38.4,"rfhi":7.6,"sst":28.2,"threat":"Low"},
        {"name":"Interview Island","region":"Andaman Nicobar","lat":13.18,"lon":92.72,"health":"Healthy","species":320,"area_km2":26.5,"rfhi":8.1,"sst":28.0,"threat":"Low"},
        {"name":"Spike Island","region":"Andaman Nicobar","lat":12.62,"lon":92.85,"health":"Bleached","species":198,"area_km2":8.3,"rfhi":3.9,"sst":30.1,"threat":"High"},
        # Lakshadweep
        {"name":"Agatti Island Reef","region":"Lakshadweep","lat":10.85,"lon":72.18,"health":"Healthy","species":289,"area_km2":27.1,"rfhi":8.4,"sst":27.5,"threat":"Low"},
        {"name":"Bangaram Atoll","region":"Lakshadweep","lat":10.95,"lon":72.27,"health":"Healthy","species":302,"area_km2":19.6,"rfhi":8.7,"sst":27.3,"threat":"Low"},
        {"name":"Kavaratti Lagoon","region":"Lakshadweep","lat":10.57,"lon":72.64,"health":"Bleached","species":215,"area_km2":16.4,"rfhi":4.2,"sst":30.6,"threat":"High"},
        {"name":"Kalpeni Reef","region":"Lakshadweep","lat":10.08,"lon":73.65,"health":"Healthy","species":261,"area_km2":14.0,"rfhi":7.9,"sst":27.8,"threat":"Low"},
        {"name":"Minicoy Atoll","region":"Lakshadweep","lat":8.28,"lon":73.05,"health":"Bleached","species":178,"area_km2":12.0,"rfhi":3.6,"sst":30.9,"threat":"High"},
        {"name":"Kadmat Island","region":"Lakshadweep","lat":11.23,"lon":72.78,"health":"Healthy","species":245,"area_km2":11.2,"rfhi":8.0,"sst":27.6,"threat":"Low"},
        {"name":"Androth Reef","region":"Lakshadweep","lat":10.82,"lon":73.68,"health":"Dead","species":52,"area_km2":5.1,"rfhi":0.8,"sst":32.1,"threat":"Critical"},
        {"name":"Amini Reef","region":"Lakshadweep","lat":11.12,"lon":72.73,"health":"Bleached","species":190,"area_km2":8.4,"rfhi":3.2,"sst":30.7,"threat":"High"},
        # Gulf of Mannar
        {"name":"Pamban Island (Rameswaram)","region":"Gulf of Mannar","lat":9.28,"lon":79.32,"health":"Bleached","species":117,"area_km2":6.2,"rfhi":3.0,"sst":30.3,"threat":"High"},
        {"name":"Vaan Island","region":"Gulf of Mannar","lat":8.98,"lon":78.20,"health":"Dead","species":38,"area_km2":2.4,"rfhi":0.5,"sst":32.4,"threat":"Critical"},
        {"name":"Tuticorin Reef","region":"Gulf of Mannar","lat":8.80,"lon":78.15,"health":"Dead","species":29,"area_km2":3.1,"rfhi":0.3,"sst":32.8,"threat":"Critical"},
        {"name":"Shingle Island","region":"Gulf of Mannar","lat":9.20,"lon":79.00,"health":"Bleached","species":142,"area_km2":4.5,"rfhi":2.6,"sst":31.0,"threat":"High"},
        {"name":"Keezhakkarai Reef","region":"Gulf of Mannar","lat":9.22,"lon":78.82,"health":"Bleached","species":130,"area_km2":3.8,"rfhi":2.9,"sst":30.8,"threat":"High"},
        {"name":"Mullai Island","region":"Gulf of Mannar","lat":9.10,"lon":78.30,"health":"Dead","species":21,"area_km2":1.9,"rfhi":0.4,"sst":33.1,"threat":"Critical"},
        # Gulf of Kutch
        {"name":"Pirotan Island MNP","region":"Gulf of Kutch","lat":22.60,"lon":70.23,"health":"Bleached","species":73,"area_km2":162.0,"rfhi":3.8,"sst":29.9,"threat":"High"},
        {"name":"Narara Reef","region":"Gulf of Kutch","lat":22.60,"lon":70.00,"health":"Bleached","species":65,"area_km2":28.0,"rfhi":3.5,"sst":29.7,"threat":"High"},
        {"name":"Marine NP Jamnagar","region":"Gulf of Kutch","lat":22.50,"lon":69.80,"health":"Bleached","species":91,"area_km2":270.0,"rfhi":4.0,"sst":29.4,"threat":"Medium"},
        {"name":"Positra Reef","region":"Gulf of Kutch","lat":22.40,"lon":70.40,"health":"Dead","species":18,"area_km2":4.5,"rfhi":0.7,"sst":32.5,"threat":"Critical"},
        # Palk Bay
        {"name":"Karaichalli Island","region":"Palk Bay","lat":10.12,"lon":79.78,"health":"Bleached","species":96,"area_km2":3.2,"rfhi":2.4,"sst":31.2,"threat":"High"},
        {"name":"Palk Bay Fringing Reef","region":"Palk Bay","lat":9.80,"lon":79.50,"health":"Dead","species":34,"area_km2":5.0,"rfhi":0.6,"sst":32.6,"threat":"Critical"},
    ]

    REGION_LABELS = {
        "Andaman Nicobar": "Andaman & Nicobar",
        "Lakshadweep": "Lakshadweep",
        "Gulf of Mannar": "Gulf of Mannar",
        "Gulf of Kutch": "Gulf of Kutch",
        "Palk Bay": "Palk Bay",
    }

    df_reefs = pd.DataFrame(REEF_SITES)
    HEALTH_RGB = {"Healthy":[16,185,129],"Bleached":[245,158,11],"Dead":[239,68,68]}
    HEALTH_HEX = {"Healthy":"#10b981","Bleached":"#f59e0b","Dead":"#ef4444"}
    df_reefs["color"] = df_reefs["health"].map(HEALTH_RGB)
    df_reefs["radius"] = df_reefs["area_km2"].apply(lambda x: max(18000, min(60000, x*900)))
    df_reefs["region_label"] = df_reefs["region"].map(REGION_LABELS)
    df_reefs["tooltip_html"] = df_reefs.apply(lambda r:
        f"<b style='color:#38bdf8'>{r['name']}</b><br/>"
        f"<span style='color:#aaa'>Region:</span> {r['region_label']}<br/>"
        f"<span style='color:#aaa'>Health:</span> <b style='color:{HEALTH_HEX[r['health']]}'>{r['health']}</b><br/>"
        f"<span style='color:#aaa'>RFHI:</span> {r['rfhi']}/10 &nbsp; SST: {r['sst']}°C<br/>"
        f"<span style='color:#aaa'>Species:</span> {r['species']} &nbsp; Area: {r['area_km2']} km²<br/>"
        f"<span style='color:#aaa'>Threat Level:</span> {r['threat']}", axis=1)

    # ── Filter controls ──────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1.2,1.2,1], gap="medium")
    with fc1:
        sel_region = st.multiselect("🗺️ Filter Region",
            ["Andaman Nicobar","Lakshadweep","Gulf of Mannar","Gulf of Kutch","Palk Bay"],
            default=["Andaman Nicobar","Lakshadweep","Gulf of Mannar","Gulf of Kutch","Palk Bay"],
            format_func=lambda x: REGION_LABELS[x], key="geo_region")
    with fc2:
        sel_health = st.multiselect("🩺 Filter Health",
            ["Healthy","Bleached","Dead"],
            default=["Healthy","Bleached","Dead"], key="geo_health")
    with fc3:
        map_style_opt = st.selectbox("🎨 Map Style",["Dark","Satellite","Light"], key="geo_style")

    MAP_STYLES = {
        "Dark":"mapbox://styles/mapbox/dark-v10",
        "Satellite":"mapbox://styles/mapbox/satellite-streets-v11",
        "Light":"mapbox://styles/mapbox/light-v10",
    }

    df_f = df_reefs[df_reefs["region"].isin(sel_region) & df_reefs["health"].isin(sel_health)].copy()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Map + Panel ──────────────────────────────────────────────────────────
    map_col, panel_col = st.columns([2.2,1], gap="large")

    with map_col:
        scatter = pdk.Layer("ScatterplotLayer", data=df_f,
            get_position="[lon,lat]", get_fill_color="color", get_radius="radius",
            opacity=0.82, stroked=True, get_line_color=[255,255,255,60],
            line_width_min_pixels=1, pickable=True)
        pulse = pdk.Layer("ScatterplotLayer", data=df_f[df_f["health"]=="Dead"],
            get_position="[lon,lat]", get_fill_color=[239,68,68,40],
            get_radius="radius", opacity=0.3, stroked=False, pickable=False)
        deck = pdk.Deck(
            map_style=MAP_STYLES[map_style_opt],
            initial_view_state=pdk.ViewState(latitude=14.0,longitude=82.0,zoom=4.2,pitch=20),
            layers=[pulse, scatter],
            tooltip={"html":"{tooltip_html}","style":{
                "backgroundColor":"#0f172a","color":"white",
                "border":"1px solid #38bdf8","borderRadius":"10px",
                "fontSize":"13px","padding":"12px","maxWidth":"280px"}})
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:1.5rem;margin-top:0.8rem;flex-wrap:wrap">
            <div style="display:flex;align-items:center;gap:0.5rem">
                <div style="width:13px;height:13px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b98180"></div>
                <span style="color:rgba(255,255,255,0.6);font-size:0.82rem">Healthy</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem">
                <div style="width:13px;height:13px;border-radius:50%;background:#f59e0b;box-shadow:0 0 8px #f59e0b80"></div>
                <span style="color:rgba(255,255,255,0.6);font-size:0.82rem">Bleached</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem">
                <div style="width:13px;height:13px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef444480"></div>
                <span style="color:rgba(255,255,255,0.6);font-size:0.82rem">Dead / Critical</span>
            </div>
            <span style="color:rgba(255,255,255,0.28);font-size:0.75rem;margin-left:auto">Hover site for details &bull; Circle size proportional to reef area</span>
        </div>""", unsafe_allow_html=True)

    with panel_col:
        total = len(df_f)
        h_c = len(df_f[df_f["health"]=="Healthy"])
        b_c = len(df_f[df_f["health"]=="Bleached"])
        d_c = len(df_f[df_f["health"]=="Dead"])
        avg_rfhi = df_f["rfhi"].mean() if total else 0
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:1rem">
            <div style="background:rgba(16,185,129,0.1);border:1px solid #10b98140;border-radius:12px;padding:0.9rem;text-align:center">
                <h2 style="color:#10b981;margin:0;font-size:1.8rem">{h_c}</h2>
                <p style="color:rgba(255,255,255,0.4);margin:0;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Healthy</p>
            </div>
            <div style="background:rgba(245,158,11,0.1);border:1px solid #f59e0b40;border-radius:12px;padding:0.9rem;text-align:center">
                <h2 style="color:#f59e0b;margin:0;font-size:1.8rem">{b_c}</h2>
                <p style="color:rgba(255,255,255,0.4);margin:0;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Bleached</p>
            </div>
            <div style="background:rgba(239,68,68,0.1);border:1px solid #ef444440;border-radius:12px;padding:0.9rem;text-align:center">
                <h2 style="color:#ef4444;margin:0;font-size:1.8rem">{d_c}</h2>
                <p style="color:rgba(255,255,255,0.4);margin:0;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Dead</p>
            </div>
            <div style="background:rgba(56,189,248,0.08);border:1px solid #38bdf840;border-radius:12px;padding:0.9rem;text-align:center">
                <h2 style="color:#38bdf8;margin:0;font-size:1.8rem">{avg_rfhi:.1f}</h2>
                <p style="color:rgba(255,255,255,0.4);margin:0;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Avg RFHI</p>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p style="color:rgba(255,255,255,0.4);font-size:0.72rem;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.5rem">📍 Research Sites (sorted by RFHI)</p>', unsafe_allow_html=True)
        threat_colors = {"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444","Critical":"#7f1d1d"}
        for _, row in df_f.sort_values("rfhi").iterrows():
            hc = HEALTH_HEX[row["health"]]
            tc = threat_colors.get(row["threat"], "#64748b")
            short_region = row["region"].split()[0]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-left:3px solid {hc};border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.35rem">
                <p style="color:white;font-size:0.81rem;margin:0 0 0.15rem;font-weight:600;line-height:1.3">{row['name']}</p>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:{hc};font-size:0.7rem;font-weight:700">{row['health']}</span>
                    <span style="color:rgba(255,255,255,0.35);font-size:0.68rem">{short_region}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.2rem">
                    <span style="color:rgba(255,255,255,0.32);font-size:0.68rem">RFHI {row['rfhi']}/10 · {row['species']} sp.</span>
                    <span style="color:{tc};font-size:0.66rem;background:{tc}18;padding:0.08rem 0.4rem;border-radius:4px">{row['threat']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Bottom stats bar ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    b1,b2,b3,b4,b5 = st.columns(5, gap="large")
    bottom_stats = [
        ("🏝️","Total Sites", str(total), "across India"),
        ("🐠","Avg Species", str(int(df_f["species"].mean())) if total else "0", "per reef site"),
        ("🌡️","Avg SST", f"{df_f['sst'].mean():.1f}°C" if total else "—", "sea surface temp"),
        ("📐","Total Area", f"{df_f['area_km2'].sum():.0f} km²" if total else "—", "reef coverage"),
        ("⚠️","Critical Zones", str(len(df_f[df_f["threat"]=="Critical"])), "need urgent action"),
    ]
    for col,(em,title,val,sub) in zip([b1,b2,b3,b4,b5], bottom_stats):
        with col:
            st.markdown(f'<div class="stat-card" style="text-align:center"><div style="font-size:1.8rem">{em}</div><h4 style="color:white;margin:0.4rem 0 0.15rem;font-size:1rem">{val}</h4><p style="color:rgba(255,255,255,0.38);font-size:0.78rem;margin:0">{title}<br><span style="font-size:0.7rem">{sub}</span></p></div>', unsafe_allow_html=True)
'''

final = keep + new_geo
with open('frontend/app.py', 'w', encoding='utf-8') as f:
    f.write(final)
print(f"Done. Total chars: {len(final)}")
