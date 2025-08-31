"""
import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    '''Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    '''

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
"""

import streamlit as st
import numpy as np
from sim.core import Circuit
from sim.components import Source, PhaseShifter, BeamSplitter, Detector


st.set_page_config(page_title="Quantum Optics Simulator (MVP)", layout="wide")


st.title("🔬 Quantum Optics Simulator — MVP")


if "components" not in st.session_state:
    st.session_state.components = []


# Sidebar: add components
st.sidebar.header("Add component")
comp_type = st.sidebar.selectbox("Type", ["Source", "PhaseShifter", "BeamSplitter", "Detector"])


params = {}
if comp_type == "Source":
    params["mode"] = st.sidebar.selectbox("Input photon mode", [0,1], index=0)
elif comp_type == "PhaseShifter":
    params["mode"] = st.sidebar.selectbox("Mode", [0,1], index=0)
    params["phi"] = st.sidebar.number_input("phi (radians)", value=0.0, step=0.1, format="%.6f")
elif comp_type == "BeamSplitter":
    params["theta"] = st.sidebar.number_input("theta (radians)", value=float(np.pi/4), step=0.05, format="%.6f")
    params["phi"] = st.sidebar.number_input("phi (radians)", value=0.0, step=0.1, format="%.6f")
else:
    pass


if st.sidebar.button("➕ Add to circuit"):
    if comp_type == "Source":
        st.session_state.components.append((comp_type, params.copy()))
    elif comp_type == "PhaseShifter":
        st.session_state.components.append((comp_type, params.copy()))
    elif comp_type == "BeamSplitter":
        st.session_state.components.append((comp_type, params.copy()))
    elif comp_type == "Detector":
        st.session_state.components.append((comp_type, {}))


st.sidebar.button("🗑️ Clear", on_click=lambda: st.session_state.update({"components": []}))


# Show current circuit
st.subheader("Circuit")
cols = st.columns(3)
with cols[0]:
    st.write("**Sequence**")
    for i,(t,p) in enumerate(st.session_state.components):
        st.write(f"{i+1}. {t} {p}")


# Build and simulate
circuit = Circuit()
for t,p in st.session_state.components:
    if t == "Source":
        circuit.add(Source(mode=p["mode"]))
    elif t == "PhaseShifter":
        circuit.add(PhaseShifter(phi=p["phi"], mode=p["mode"]))
    elif t == "BeamSplitter":
        circuit.add(BeamSplitter(theta=p["theta"], phi=p["phi"]))
    elif t == "Detector":
        circuit.add(Detector())


if st.button("▶️ Run simulation"):
    out = circuit.simulate()
    psi = out["state_out"]
    U = out["unitary"]
    res = out["results"]

    st.subheader("Results")
    st.write("**Output state:** ψ_out")
    st.code(np.array2string(psi, precision=5))

    st.write("**Global unitary (U):**")
    st.code(np.array2string(U, precision=5))

    st.metric("p(mode0)", f"{res['p(mode0)']:.4f}")
    st.metric("p(mode1)", f"{res['p(mode1)']:.4f}")


# Simple MZI sanity demo button
st.divider()
st.markdown("### Quick demo: Mach–Zehnder with controllable phase")
phi = st.slider("Internal phase φ (radians)", min_value=0.0, max_value=float(2*np.pi), value=0.0, step=0.01)

mzi = Circuit()
mzi.add(Source(mode=0))
mzi.add(BeamSplitter(theta=np.pi/4))
mzi.add(PhaseShifter(phi=phi, mode=0))
mzi.add(BeamSplitter(theta=np.pi/4))
mzi.add(Detector())

out = mzi.simulate()
st.write(out["results"]) # shows interference vs φ
