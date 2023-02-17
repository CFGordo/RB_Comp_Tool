
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import agstyler
from agstyler import PINLEFT, PRECISION_TWO, draw_grid, draw_grid2
from enum import Enum

st.title("🏈 🏈 CFB RB Comp Tool 🏈 🏈")
st.text("")
st.text("")
rb_comp_df_link = st.secrets["rb_comp_df_url"]
rb_comp_df_csv = rb_comp_df_link.replace('/edit#gid=', '/export?format=csv&gid=')
rb_comp_df = pd.read_csv(rb_comp_df_csv)

rb_comp_df.sort_values(by=['Season', 'model_score'], ascending=[False, False], inplace=True)

player1 = st.selectbox('Select PLAYER!', rb_comp_df['Name'].unique())

player_comp_df = rb_comp_df.loc[rb_comp_df['Name'] == player1]
st.text("")


class Color(Enum):
    GREEN_LIGHT = "rgb(0, 120, 60, .6)"
    RED_LIGHT = "rgb(120, 0, 40, .5)"


condition_one_value = "params.value >= 9"
condition_two_value = "params.value < 5"

formatter = {
    'Season': ('Year', {'width': 60}),
    'Player ID': ('Player ID', {'width': 70}),
    'Team': ('Team', {'width': 55}),
    'Games Played': ('Games Played', {'width': 60, 'cellStyle': agstyler.highlight(Color.RED_LIGHT.value,
                                                                                   condition_two_value)}),
    'Rushing Efficiency': ('Rushing Efficiency', {**PRECISION_TWO, 'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Receiving Efficiency': ('Receiving Efficiency', {**PRECISION_TWO, 'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Rushing Explosiveness': ('Rushing Explosiveness', {**PRECISION_TWO, 'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Receiving Explosiveness': ('Receiving Explosiveness', {**PRECISION_TWO, 'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
}
st.markdown("Select Season to Compare:")

data = draw_grid(
    player_comp_df[['Season', 'Player ID', 'Team', 'Games Played', 'Rushing Efficiency', 'Receiving Efficiency',
                    'Rushing Explosiveness', 'Receiving Explosiveness']].head(),
    formatter=formatter,
    fit_columns=True,
    selection='single',  # or 'single', or None
    use_checkbox='True',  # or False by default
    max_height=400
)
selected = data["selected_rows"]
select2 = pd.DataFrame(selected)

if selected:
    player2 = int(select2['Player ID'])
    player3 = int(select2['Season'])
else:
    player4 = player_comp_df.loc[player_comp_df['Season'] == player_comp_df['Season'].max()]
    player8 = player4['Season'].unique()
    player3 = int(player8)
    player11 = player4.loc[player4['Player ID'] == player4['Player ID'].min()]
    player2 = int(player11['Player ID'])
playerSZN_comp_df = player_comp_df.loc[player_comp_df['Player ID'] == player2]
playerSZN_comp_df = playerSZN_comp_df.loc[playerSZN_comp_df['Season'] == player3]

riker = {"Criteria": ["Height", "Weight", "Rushing Efficiency", "Rushing Explosiveness", "Receiving Efficiency",
                      "Rec Explosiveness", "Receiving Best", "Team Talent", "Team SP Rating", "NFL Draft Position"],
         "Recomended?": ["Strongly", "Strongly", "Strongly", "Strongly", "Strongly",
                         "Strongly", "*Instead of* Receiving Efficiency & Explosiveness",
                         "Yes, if *Not* using SP", "Yes, if *Not* using Tm Talent", "Post Draft Only"],
         "Notes": ["as listed by school", "as listed by school",
                      "Success Rate with adjustments for Weight and Volume. Scaled and Capped at 20.",
                      "Yards per Carry scewed heavily towards Big Runs (>=12yds) adjusted for Volume. Scaled and Capped at 20.",
                      "Succes Rate by Target with adjustments for Volume. Scaled and Capped at 20.",
                      "Yards per Reception scewed heavily towards Big Recs (>=12yds) adjusted for Volume. Scaled and "
                      "Capped at 20.", "Max of Receiving Efficiency & Explosiveness", "per 247 Sports", "per Bill Connelly",
                      "`Draft Cap isn't Everything; It's the Only thing.` - Vince Lombardi - Don't "
                      "use it, coward."]}

criteria = pd.DataFrame(data=riker)
fromage = {
    'Criteria': ('Criteria', {'width': 50}),
    'Recomended?': ('Recomended?', {'width': 50}),
    'Notes': ('Notes', {'width': 100})
}
st.text("")
st.markdown("Select Criteria:")
datas = draw_grid2(
    criteria,
    formatter=fromage,
    fit_columns=True,
    selection='multiple',  # or 'single', or None
    pre_selected_rows=[0, 1, 2, 3, 4, 5, 7],
    use_checkbox='True',  # or False by default
    auto_height=True,
    max_height=800
)
selection = datas["selected_rows"]
selectionz = pd.DataFrame(selection)
selecto = {"Criteria": ["unk"], "Recomended?": ["Yaas Queen"], "Notes": ["Nicky"]}
selector = pd.DataFrame(selecto)
selections = pd.concat([selector, selectionz])

Height = (selections['Criteria'].eq('Height')).any()
Weight = (selections['Criteria'].eq('Weight')).any()
Rush_Eff = (selections['Criteria'].eq('Rushing Efficiency')).any()
Rush_Exp = (selections['Criteria'].eq('Rushing Explosiveness')).any()
Pass_Eff = (selections['Criteria'].eq('Receiving Efficiency')).any()
Pass_Exp = (selections['Criteria'].eq('Rec Explosiveness')).any()
Pass_Best = (selections['Criteria'].eq('Receiving Best')).any()
Talent = (selections['Criteria'].eq('Team Talent')).any()
SP = (selections['Criteria'].eq('Team SP Rating')).any()
Pick = (selections['Criteria'].eq('NFL Draft Position')).any()

if Height:
     rb_comp_df['comp_height_comp'] = rb_comp_df['comp_height']-playerSZN_comp_df['comp_height'].max()
     rb_comp_df['comp_height_compass'] = np.where(rb_comp_df['comp_height_comp'] < 0, -1, 1)
     rb_comp_df['comp_height_comps'] = rb_comp_df['comp_height_comp'] * rb_comp_df['comp_height_compass']
else:
    rb_comp_df['comp_height_comps'] = 0
if Weight:
     rb_comp_df['comp_weight_comp'] = rb_comp_df['Weight']-playerSZN_comp_df['Weight'].max()
     rb_comp_df['comp_weight_compass'] = np.where(rb_comp_df['comp_weight_comp'] < 0, -1, 1)
     rb_comp_df['comp_weight_comps'] = rb_comp_df['comp_weight_comp'] * rb_comp_df['comp_weight_compass']
else:
    rb_comp_df['comp_weight_comps'] = 0
if Rush_Eff:
     rb_comp_df['comp_rusheff_comp'] = rb_comp_df['Rushing Efficiency']-playerSZN_comp_df['Rushing Efficiency'].max()
     rb_comp_df['comp_rusheff_compass'] = np.where(rb_comp_df['comp_rusheff_comp'] < 0, -1, 1)
     rb_comp_df['comp_rusheff_comps'] = rb_comp_df['comp_rusheff_comp'] * rb_comp_df['comp_rusheff_compass']
else:
    rb_comp_df['comp_rusheff_comps'] = 0
if Rush_Exp:
     rb_comp_df['comp_rushexp_comp'] = rb_comp_df['Rushing Explosiveness']-playerSZN_comp_df['Rushing Explosiveness'].max()
     rb_comp_df['comp_rushexp_compass'] = np.where(rb_comp_df['comp_rushexp_comp'] < 0, -1, 1)
     rb_comp_df['comp_rushexp_comps'] = rb_comp_df['comp_rushexp_comp'] * rb_comp_df['comp_rushexp_compass']
else:
    rb_comp_df['comp_rushexp_comps'] = 0
if Pass_Eff:
     rb_comp_df['comp_passeff_comp'] = rb_comp_df['Receiving Efficiency']-playerSZN_comp_df['Receiving Efficiency'].max()
     rb_comp_df['comp_passeff_compass'] = np.where(rb_comp_df['comp_passeff_comp'] < 0, -1, 1)
     rb_comp_df['comp_passeff_comps'] = rb_comp_df['comp_passeff_comp'] * rb_comp_df['comp_passeff_compass']
else:
    rb_comp_df['comp_passeff_comps'] = 0
if Pass_Exp:
     rb_comp_df['comp_passexp_comp'] = rb_comp_df['Receiving Explosiveness']-playerSZN_comp_df['Receiving Explosiveness'].max()
     rb_comp_df['comp_passexp_compass'] = np.where(rb_comp_df['comp_passexp_comp'] < 0, -1, 1)
     rb_comp_df['comp_passexp_comps'] = rb_comp_df['comp_passexp_comp'] * rb_comp_df['comp_passexp_compass']
else:
    rb_comp_df['comp_passexp_comps'] = 0
if Pass_Best:
     rb_comp_df['comp_passbest_comp'] = rb_comp_df['Receiving Best']-playerSZN_comp_df['Receiving Best'].max()
     rb_comp_df['comp_passbest_compass'] = np.where(rb_comp_df['comp_passbest_comp'] < 0, -1, 1)
     rb_comp_df['comp_passbest_comps'] = rb_comp_df['comp_passbest_comp'] * rb_comp_df['comp_passbest_compass']
else:
    rb_comp_df['comp_passbest_comps'] = 0
if Talent:
     rb_comp_df['comp_talent_comp'] = rb_comp_df['comp_talent']-playerSZN_comp_df['comp_talent'].max()
     rb_comp_df['comp_talent_compass'] = np.where(rb_comp_df['comp_talent_comp'] < 0, -1, 1)
     rb_comp_df['comp_talent_comps'] = rb_comp_df['comp_talent_comp'] * rb_comp_df['comp_talent_compass']
else:
    rb_comp_df['comp_talent_comps'] = 0
if SP:
     rb_comp_df['comp_sp_comp'] = rb_comp_df['comp_sp']-playerSZN_comp_df['comp_sp'].max()
     rb_comp_df['comp_sp_compass'] = np.where(rb_comp_df['comp_sp_comp'] < 0, -1, 1)
     rb_comp_df['comp_sp_comps'] = rb_comp_df['comp_sp_comp'] * rb_comp_df['comp_sp_compass']
else:
    rb_comp_df['comp_sp_comps'] = 0
if Pick:
     rb_comp_df['comp_pick_comp'] = rb_comp_df['comp_ovr']-playerSZN_comp_df['comp_ovr'].max()
     rb_comp_df['comp_pick_compass'] = np.where(rb_comp_df['comp_pick_comp'] < 0, -1, 1)
     rb_comp_df['comp_pick_comps'] = rb_comp_df['comp_pick_comp'] * rb_comp_df['comp_pick_compass']
else:
    rb_comp_df['comp_pick_comps'] = 0

st.markdown("")
st.markdown("The Comps:")
rb_comp_df['comp_scor'] = rb_comp_df[['comp_talent_comps', 'comp_height_comps',
                                      'comp_weight_comps', 'comp_rushexp_comps', 'comp_rusheff_comps',
                                      'comp_passexp_comps', 'comp_passeff_comps', 'comp_sp_comps',
                                      'comp_passbest_comps', 'comp_pick_comps']].sum(axis=1)

rb_comp_df['Comp Score'] = 100-rb_comp_df['comp_scor']


formatter2 = {
    'Name': ('Name [select to compare]', {**PINLEFT, 'width': 50}),
    'Comp Score': ('Score', {**PRECISION_TWO, 'width': 50}),
    'Team': ('Team', {'width': 40}),
    'Season': ('Year', {'width': 45}),
    'Games Played': ('G', {'width': 5}),
    'Height': ('Ht', {'width': 30}),
    'Weight': ('Wt', {'width': 40}),
    'Rushing Efficiency': ('Rushing Efficiency', {'width': 70, 'cellStyle': agstyler.highlight(
        Color.GREEN_LIGHT.value, condition_one_value)}),
    'Receiving Efficiency': ('Receiving Efficiency', {'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Rushing Explosiveness': ('Rushing Explosiveness', {'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Receiving Explosiveness': ('Receiving Explosiveness', {'width': 70, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                                                condition_one_value)}),
    'Team Talent': ('Team Talent', {'width': 60}),
    'SP Rating': ('SP Rating', {'width': 60}),
    'Draft Year': ('Draft Class', {'width': 60}),
    'NFL Draft Pick': ('NFL Draft Pick', {'width': 60}),
    'NFL PPR PPG': ('NFL PPR PPG', {'width': 60, 'cellStyle': agstyler.highlight(Color.GREEN_LIGHT.value,
                                                                                 condition_one_value)}),
    'Player ID': ('ID', {'width': 50})
}

games = st.slider('Games Played Filter', 1, 15, 4)
nfl_comp = rb_comp_df.loc[rb_comp_df['Games Played'] >= games]
nfl_only = st.checkbox('NFL Only')
if nfl_only:
    nfl_comp = nfl_comp.loc[nfl_comp['NFL Draft Pick'] > 0]
    nfl_comp['max_comp'] = nfl_comp.groupby(['Player ID'])['Comp Score'].transform('max')
    nfl_comp = nfl_comp.loc[nfl_comp['max_comp'] == nfl_comp['Comp Score']]
    playerNFL_comp_df = rb_comp_df.loc[rb_comp_df['Player ID'] == player2]
    playerNFL_comp_df = playerNFL_comp_df.loc[playerNFL_comp_df['Season'] == player3]
    nfl_comp2 = pd.concat([nfl_comp, playerNFL_comp_df])
    nfl_comp2 = nfl_comp2[['Name', 'Comp Score', 'Team', 'Season', 'Games Played', 'Height', 'Weight', 'Rushing Efficiency',
                           'Receiving Efficiency', 'Rushing Explosiveness', 'Receiving Explosiveness', 'Team Talent',
                           'SP Rating', 'Draft Year', 'NFL Draft Pick', 'NFL PPR PPG', 'Player ID']]
else:
    nfl_comp2 = nfl_comp[['Name', 'Comp Score', 'Team', 'Season', 'Games Played', 'Height', 'Weight', 'Rushing Efficiency',
                           'Receiving Efficiency', 'Rushing Explosiveness', 'Receiving Explosiveness', 'Team Talent',
                           'SP Rating', 'Draft Year', 'NFL Draft Pick', 'NFL PPR PPG', 'Player ID']]
row_number = st.number_input('Number of Comps', min_value=0, value=11)
data2 = draw_grid(
    nfl_comp2.loc[nfl_comp2['Games Played'] >= 4].sort_values(by=['Comp Score'], ascending=False).round(decimals=1).head(row_number),
    formatter=formatter2,
    fit_columns=False,
    selection='multiple',  # or 'single', or None
    use_checkbox='True',  # or False by default
    max_height=350,
    auto_height=True
    )
st.caption("Hint: You can sort and filter columns to find your own Comps. Increase 'Number of "
           "Comps' above the table to get more players.")
cell = data2["selected_rows"]
cellz = pd.DataFrame(cell)
cellar = rb_comp_df.loc[rb_comp_df['Player ID'] == player2]
cellars = pd.concat([cellar, cellz])
celly = cellars['Player ID'].drop_duplicates()
cellary = pd.merge(celly, rb_comp_df, on='Player ID', how='left')

measures = ['Rush Success Rate', 'Succ Rushes perG', 'Big Rush Rate', 'Big Rushes perG',
            'Big Rush Yards perG', 'Target Success Rate',  'Succ Targs perG', 'Big Rec Rate', 'Big Recs perG',
            'Big Rec Yards perG', 'YTMRA perG', 'YTMPA perG', 'YTMA perG',
            'Rush PPA perG', 'Rec PPA perG', 'Total PPA perG', 'PPA per Rush Att', 'PPA per Target',
            'PPA per Opportunity']
pleasures = ['Rush Success Rate (successful rush per attempt)', 'Successful Rushes per Game',
             'Big Rush Rate (rush >= 12yds per attempt)', 'Big Rushes per Game (rushes >= 12yds)',
             'Big Rush Yards per Game (yds on big rushes, not including first 12 yds)',
             'Target Success Rate (successful reception per target)',
             'Successful Targets per Game', 'Big Reception Rate (reception >=12yds per reception)',
             'Big Rececptions per Game (receptions >= 12yds)',
             'Big Reception Yards per Game (yds on big receptions, not including first 12 yds)',
             'Rush Yards per Team Rush Attempt per Game', 'Reception Yards per Team Pass Attempt per Game',
             'Total Yards per Team Offensive Play per Game',
             'Rush PPA per Game (version of EPA, per collegefootballdata.com)',
             'Receiving PPA per Game (version of EPA, per collegefootballdata.com)',
             'Total PPA per Game (version of EPA, per collegefootballdata.com)',
             'PPA per Rush Att (version of EPA, per collegefootballdata.com)',
             'PPA per Target(version of EPA, per collegefootballdata.com)',
             'PPA per Opportunity (version of EPA, per collegefootballdata.com)']

stit = st.selectbox('Select Stat!', pleasures)

if stit == 'Rush Success Rate (successful rush per attempt)':
    stat = 'Rush Success Rate'
if stit == 'Successful Rushes per Game':
    stat = 'Succ Rushes perG'
if stit == 'Big Rush Rate (rush >= 12yds per attempt)':
    stat = 'Big Rush Rate'
if stit == 'Big Rushes per Game (rushes >= 12yds)':
    stat = 'Big Rushes perG'
if stit == 'Big Rush Yards per Game (yds on big rushes, not including first 12 yds)':
    stat = 'Big Rush Yards perG'
if stit == 'Target Success Rate (successful reception per target)':
    stat = 'Target Success Rate'
if stit == 'Successful Targets per Game':
    stat = 'Succ Targs perG'
if stit == 'Big Reception Rate (reception >=12yds per reception)':
    stat = 'Big Rec Rate'
if stit == 'Big Rececptions per Game (receptions >= 12yds)':
    stat = 'Big Recs perG'
if stit == 'Big Reception Yards per Game (yds on big receptions, not including first 12 yds)':
    stat = 'Big Rec Yards perG'
if stit == 'Rush Yards per Team Rush Attempt per Game':
    stat = 'YTMRA perG'
if stit == 'Reception Yards per Team Pass Attempt per Game':
    stat = 'YTMPA perG'
if stit == 'Total Yards per Team Offensive Play per Game':
    stat = 'YTMA perG'
if stit == 'Rush PPA per Game (version of EPA, per collegefootballdata.com)':
    stat = 'Rush PPA perG'
if stit == 'Receiving PPA per Game (version of EPA, per collegefootballdata.com)':
    stat = 'Rec PPA perG'
if stit == 'Total PPA per Game (version of EPA, per collegefootballdata.com)':
    stat = 'Total PPA perG'
if stit == 'PPA per Rush Att (version of EPA, per collegefootballdata.com)':
    stat = 'PPA per Rush Att'
if stit == 'PPA per Target(version of EPA, per collegefootballdata.com)':
    stat = 'PPA per Target'
if stit == 'PPA per Opportunity (version of EPA, per collegefootballdata.com)':
    stat = 'PPA per Opportunity'
else:
    stat = 'Rush Success Rate'

statable = cellary[['Year from Highschool', 'Name', 'Player ID', stat]]
statable['Year from Highschool'] = statable['Year from Highschool'].astype(int)
statable['Name'] = statable['Name'].astype(str)
statable['Player ID'] = statable['Player ID'].astype(int)
statable[stat] = statable[stat].astype(float)
st.table(statable[stat])
restatable = pd.pivot_table(statable, values=stat, index=['Name', 'Player ID'],
                            columns=['Year from Highschool'], aggfunc=np.sum)
restatable.reset_index(level=1, inplace=True)
restatable.drop(columns={'Player ID'}, inplace=True)
stable = restatable.transpose()

mess = rb_comp_df.loc[rb_comp_df['Year from Highschool'] <= statable['Year from Highschool'].max()]
mess.replace([np.inf, -np.inf], np.nan, inplace=True)
messy = mess.loc[mess['Games Played'] >= 8]
messy = messy.loc[messy['Big Rec Rate'] < 0.6]
messy = messy.loc[messy['Big Rush Rate'] < 0.4]


def interactivePlot2():
    plot = px.scatter(stable, x=None, y=None, template='simple_white')
    plot.update_traces(connectgaps=True)
    plot.update_layout(
        xaxis_title="Year from Highschool",
        yaxis_title=stat,
        legend_title="Name [select from table above]",
        plot_bgcolor="rgb(0,0,0)",
        paper_bgcolor="rgb(0,0,0)"
    )
    plot.update_traces(marker={'size': 12})
    plot.add_trace(px.scatter(messy.loc[messy['NFL Draft Pick'] > 0].dropna(), x='Year from Highschool', y=stat,
                              template='simple_white').update_layout(plot_bgcolor="rgb(0,0,0)",
                                                                     paper_bgcolor="rgb(0,0,0)").update_traces(marker={
                                                                                                        'size': 2,
                                                                                                        'color': 'gray'}).data[0])
    st.plotly_chart(plot)


st.text(rb_comp_df['NFL Draft Pick'].dtypes)


interactivePlot2()
st.text("")
st.caption("Data= collegefootballdata.com, cfbfastR, nflverse")
st.caption("Author= @CFGordon")
