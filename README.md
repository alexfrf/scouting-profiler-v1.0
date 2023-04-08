![png](/Documentacion/logomin.png)

# Segmentation & Profiling of Football Players through Clustering, Advanced Metrics and Analysis of Game Model

[Portfolio](https://alexfrf.github.io/) | [GitHub Repo](https://github.com/alexfrf/scouting-profiler)

- **Disclaimer 1: Both data and the app itself are written in Spanish**
- **Disclaimer 2: This model, including the app build-up, is entirely written in Python. You can check the code line by line directly on the repo.**


The purpose of this text is to describe a project for planning, developing, conceptualizing, and subsequently deploying a web application that visualizes a model for segmenting the characteristics of players in the world's most important football competitions according to advanced metrics. By segmenting the game model of teams, the application aims to associate the footballers who best fit into their system for each position on the field. The goal is to add the variant of the team's style of play, both their own and others, by measuring and quantifying it to find those players who are closest, based on advanced metrics, to the numbers of the team being analyzed. These footballers will potentially best suit the needs for which the team goes to the market, as they will require a shorter adaptation process and will know similar game mechanisms since they come from teams that show tactical similarities.


**App built and shared via Streamlit | [LINK](https://alexfrf-mlscoutingprofiler.streamlit.app/)**

![png](/Documentacion/streamlit.png)


#### Data Sources

*InStat* | *Transfermarkt*

In the following lines, we will explain the technical procedures that were carried out, from the extraction of the necessary data to the deployment of the application in the Streamlit cloud. We will also detail the scripts and functions used in each step. The structure of this explanation is as follows:

- **SOURCE DATA: Content and Treatment (Extraction, Import and Transformation)**
     - Instant Data
     - Transfermark data
     - Cleaning and treatment of tables + auxiliary functions
     
          
- **DATA MODELING**
     - Feature Engineering
     - Creation and Definition of Metrics
     
           
- **SEGMENTATION PROCESS**
     - Segmenting the teams + definition of key metrics
     - Segmenting the players + definition of key metrics
     

- **DISTANCE ALGORITHMS**
     - Team_Mapping: degree of adequacy of the player to a certain game model and position.
     - Player_Similarities: degree of resemblance to a given player.
     
            
- **THE APP EXPLAINED**
     - Application coding
     - Deployment and repository
     - Displayable Elements


### Source Data | Extraction, Import and Transformation

The data comes from two sources -Instat and Transfermarkt-, whose export from its origin presents differences. The Instat data has been downloaded, in excel format, from the provider's website. For each league we found two files, one corresponding to players and the other to teams. We have this information for the first divisions of 24 countries and the second divisions of the five major leagues -Spain, England, France, Italy and Germany-.

| Código | País | Liga |
|---|---|---|
| A1 | Austria | Bundesliga |
| AR1N | Argentina | PrimeraDivision |
| BE1 | Belgium | JupilerProLeague |
| BRA1 | Brazil | BrasileiroSerieA |
| BU1 | BulgariaFirstLeague | BulgariaFirstLeague |
| C1 | Switzerland | CreditSuisseSuperLeague |
| ES1 | Spain | PrimeraDivision |
| ES2 | Spain | SegundaDivision |
| FR1 | France | Ligue1 |
| FR2 | France | Ligue2 |
| GB1 | England | PremierLeague |
| GB2 | England | Championship |
| GR1 | Greece | SuperLeague |
| IT1 | Italy | SerieA |
| IT2 | Italy | SerieB |
| KR1 | Croatia | HNL |
| L1 | Germany | Bundesliga |
| L2 | Germany | Bundesliga |
| NO1 | Norway | Eliteserien |
| PL1 | Poland | PKOBankPolskiEkstraklasa |
| PO1 | Portugal | PrimeiraLiga |
| RO1 | Romania | Superliga |
| RU1 | Russia | PremierLeague |
| SC1 | Scotland | PremierLeague |
| SE1 | Sweden | Allsvenskan |
| SL1 | Slovenia | SNL |
| TR1 | Turkey | SuperLig |
| TS1 | CzechRepublic | FortunaLeague |
| UNG1 | Hungary | NBI |

On the other hand, the transfermarkt data referring to the report is scraped directly from the web using the Requests and BeautifulSoup libraries, using the code included in the *Tmarkt_scraper.py* script. This task results in the creation of three files:

- **tmarkt**: made up by the basic information of the players that are part of the competitions listed above.
- **leagues**: metadata of teams and leagues
- **equipos_sistemas**: stands for detailed information on match plans, collected in Transfermarkt [here](https://www.transfermarkt.com/Real%20Betis%20Sevilla/spielplan/verein/150/saison_id/2021/plus/1#ES1). The coach who directed the team in each match and the playing system are collected. Based on this information, a series of metrics related to the absolute and relative time of use of each structure are calculated.

Based on this information, the *data_cleaning.py* script imports the files and, after concatenating the Instat team and player data by competition, performs the following operations:


**Squads' data**

- It is not possible to connect the information from both sources directly, since the equipment does not have the same name in origin. For this reason, in order to be able to join both tables, we will use a connection algorithm by text similarities, Fuzzywuzzy, for the clubs in each competition. In this way, we will discern which Instat team each Transfermarkt ID corresponds to.

- After this process, and once each team has been assigned its identifier, we will be able to incorporate Transfermarkt data regarding leagues and game systems.


**Players' Data**

- We repeat the connection procedure between both data sources, although, in this case, given the existing complexity in the text and the existing number of lines, it is not reduced to just one iteration with the similarity algorithm.

- After this procedure, we take the resulting values ​​with a sufficient score to ensure their reliability. We will assign these names to a Transfermarkt ID. With the rest -bad score or duplicate assignments- we proceed to look for their transfermarkt ID through queries on their page in google. For those with which a positive search is found, as long as they do not show duplications with respect to the assignment previously made, an identifier will be assigned. Otherwise, the ID must be assigned manually.

- Having made the assignment of the excel identifier to unassigned elements, we will import it and join the rest of the data. We will review, before moving forward, possible duplicate cases.

- After this process, and once each team has been assigned its identifier, we will be able to incorporate the Transfermarkt data regarding players -club, market value or position according to the web-.

It should be noted that, from the moment of data export, both the team and player variables are normalized to 90 minutes.


### Data Modeling

The following notebooks are used in this process:

- *modeling_functions.py* contains the metric calculation functions 
- *modeling.py* includes the execution of the necessary functions included in the script commented above + a series of feature engineering procedures that we will explain below.

In this last script, as we said, the necessary procedures are carried out for the dimensional reduction of rows in the player dataset, in order not to include in the model players who, due to having played a reduced amount of game time or this has always been carried out in the same context -the last minutes of the match-, can distort the results and validity of the same. It is carried out using the outlier detection method via the interquartile range, based on two columns: % of matches in the competition played by the player and % of matches played in which they have entered from the bench. In addition, we will not use players who have played less than seven games for the analysis.

Subsequently, the execution of the metric calculation functions is carried out, both for players and teams, and for determining the position on the field. We complete this part by calculating a Gini index for the teams, which indicates the range of dispersion -or decentralization- in the generation of opportunities -we do it, specifically, on the variable of Goals + Expected Assists-, and, additionally, through the OneHotEncoder method, we transform the good leg variable into a numeric one. This helps us to determine, for wingers and wingers, if they act with their natural leg or not.

In addition, a function that acts on the numerical variables will be applied to both datasets, adjusting them to the possession of the teams.


### Segmentation Process

The following notebooks are used in this process:

- *clustering_functions.py* contains the functions.
- *clustering.py* includes the execution of the necessary functions included in the script commented above + a series of exports in csv of metadata necessary for the execution of the distance functions.

The model is based on a logic of distances based on the segmentation of the game models of the teams - based on variables labeled according to four very different game categories that we determine below - and the characteristics of the players in each position in the game. field.

**Game Categories for Teams Game Model**

The variables that will be assessed in order to segment the game model of the **teams** are defined below. Said analysis will be undertaken differentiating between four categories:

- Tactical disposition, understood as the organization of the team on the field
- Defense and Pressure
- Buildup / Ball possession
- Chances Creation and Definition

The function that follows will perform the clustering of the equipment based on the four categories mentioned above. Below we will analyze the results in order to outline each of the clusters that are generated for all the categories.

First, the dataset will take the columns mentioned above for the category in question, which will be normalized via the MinMaxScaler. The VarianceThreshold algorithm will be applied to it, in order to detect if there are constant or practically constant variables. After that, **PCA** will be applied up to an explained variance of 90%. The elbow method will help us determine the optimal number of clusters that the **KMeans** clustering algorithm. We will specifically mark that the model does not segment each category into more than four groups.

With this, we will obtain the clustering of the game models based on these categories. However, in order to analyze it more clearly, we will extract the most explanatory real variables (up to 90%) using a **RandomForestClassifier**. The function will return different plots that will help us in our analysis, details of the operation of the training process of the PCA and ElbowMethod processes, and a dataset with the assignment to the clusters. In addition, we will also have a dataframe with the variability explained by each of the variables.

**Tactical Disposition**

This chapter deals with the structure proposed on the field of play, referring to the system used and the number of players occupying each line, expressed as the percentage of time over the total minutes played. Next we check the weight of each variable in the model. The following stand out:

    - % of time spent forming with defense of four men
    - % of time spent training with double pivot
    - % of time spent forming with defense of four men
    - % of time spent forming with just one striker
    - % of time spent forming with three forwards

The PCA algorithm selects the first six of the present image.

![png](/Model/Teams/Feature_scores_disposicion_tactica.png)

- C1: Four defenders, double pivot, one striker - 4-2-3-1

![png](/Model/Teams/ttc1.PNG)

- C2: Four defenders, two forwards - 4-3-1-2, 4-4-2

![png](/Model/Teams/ttc2.PNG)

- C3: Three center backs, do not play with a striker - 3-4-3, 3-5-2

![png](/Model/Teams/ttc3.PNG)

- C4: Four defenders, three forwards - 4-3-3
    
![png](/Model/Teams/ttc4.PNG)    

![png](/Model/Teams/PCA6_4clusters_disposicion_tactica.png)

**Defense**

This chapter addresses the team's defensive proposal, covering variables that try to explain defensive intensity, heights and the time dedicated to being reactive.

    - Pressures for every 100 defensive actions
    - Recoveries in rival field
    - pressures
    - % of the recoveries that are made in the rival field
    - % of effective pressures
    - Rival passes by defensive action (PPDA)
    
The PCA function selects the first nine variables from this image.

![png](/Model/Teams/Feature_scores_defensa.png)

- C1: Less pressure tendency, more low withdrawal and more time defending.

![png](/Model/Teams/tdc1.PNG)

- C2: More defensive intensity in intermediate zones.

![png](/Model/Teams/tdc2.PNG)

- C3: Different heights depending on the situation and efficiency when they decide to defend high.

![png](/Model/Teams/tdc3.PNG)

- C4: More prompt to press, high defense and aggressiveness in the opposite field.

![png](/Model/Teams/tdc4.PNG)

![png](/Model/Teams/PCA9_4clusters_defensa.png)

**Buildup**

This chapter deals with the team's proposal with the ball, showing variables that try to illustrate the way in which the team proceeds with the ball out and settles and/or progresses in the rival field.

    - Incursions into rival field for every 100 passes
    - Pace - time (seconds) with the ball for each tackle in the final third
    - Buildup situations per 100 possessions
    - Time with the ball (in seconds)
    - Incursions to final third
    
The PCA function selects the first seven variables from this image.

![png](/Model/Teams/Feature_scores_buildup.png)

- C1: Long possessions, low volume of play in the final third, long transitions and from afar, little incidence of counterattack.

![png](/Model/Teams/tbc1.PNG)

- C2: Progression towards a more direct rival field, high tendency to counterattack, high pace in construction and arrival in the box with few passes.

![png](/Model/Teams/tbc2.PNG)

- C3: More elaborate ball release from the back, slow pace under construction and long possessions, little recourse to counterattack.

![png](/Model/Teams/tbc3.PNG)

- C4: Short transitions in the opposite field, relatively elaborate ball release, high volume of play in the final third, very high pace and verticality.

![png](/Model/Teams/tbc4.PNG)

![png](/Model/Teams/PCA7_4clusters_buildup.png)

**Chances Creation and Finishing**

This chapter looks at the way the team proceeds in the final third of the field, its clear-sightedness in generating opportunities and the value produced in the area.

    - Goal plays for every 100 incursions into the rival field
    - Chances created (shot or key pass) every 100 raids in the last third
    - Expected goals per 100 actions
    - Key passes per 100 passes
    - Raids in rival area
    
The PCA function selects the first eight variables from this image.

![png](/Model/Teams/Feature_scores_creacion_oportunidades.png)

- C1: Low transformation index (shot or opportunity) when reaching the rival area, reduced association to generate -greater dependence on individual action-.

![png](/Model/Teams/toc1.PNG)

- C2: Little ability to make their possessions profitable but a relatively high number of arrivals and possibilities of passing to the area/centre, low level of association in final meters.

![png](/Model/Teams/toc2.PNG)

- C3: High transformation rate (shot or opportunity) when reaching the rival area, individual quality to find a good opportunity, greater association between attacking players.

![png](/Model/Teams/toc3.PNG)

- C4: Decent transformation ratio for each attack, less association between attackers to find the opportunity.

![png](/Model/Teams/toc4.PNG)

![png](/Model/Teams/PCA8_4clusters_creacion_oportunidades.png)


**Positional categories for players**

We will segment the **players** according to their characteristics, differentiating by the position they are placed regularly on the field, making up five different groups:

- Centre-Backs (DFC)
- Full-Backs
    - Left-Backs (LI)
    - Right-backs (LD)
- Midfielders
    - Pivots and defensive midfielders (MCD)
    - Central and interior midfielders (MC)
- Advanced Midfielders or Wingers
    - Advanced Playmakers (MCO)
    - Right wingers (ED)
    - Left wingers (EI)
- Center Forwards / Strikers (DC)

The function that segments the players will perform the clustering of the players based on all the positions mentioned above. Each main position has, in turn, some key variables. Below we will analyze the results in order to outline each of the clusters that are generated for all positions.

First of all, the dataset will take the columns mentioned above for the position in question, which will be normalized via MinMaxScaler. The VarianceThreshold algorithm will be applied to it, in order to detect if there are constant or practically constant variables. The training process will begin there, which consists of applying PCA up to an explained variance of 90% and applying KMeans.

After this process, we will measure the correlation between the variables and we will discard those that present a high value in the matrix. After that, the most explanatory real variables are extracted with respect to the clusters given by the kmeans applied by means of a RandomForestClassifier, in order to clarify the model even more.

PCA is applied again, this time on the subset of variables extracted from the classifier -as many principal components as necessary to explain 90% of the variance-. After that, we generate the clusters via KMeans on the resulting PCAs.

With this, we will obtain the clustering of the attributes of the football players based on their position. The function will return different plots that will help us in our analysis, details of the operation of the training process of the PCA and ElbowMethod processes, and a dataset with the assignment to the clusters. In addition, we will also have a dataframe with the variability explained by each of the columns.

**Centre-Backs**

For this position, the most explanatory variables and those that contribute to a greater extent to compose the differences between players and group them are the following.

![png](/Model/Players/dfc_seg.png)

- C1: Greater tendency to make tackles and go out of the area -higher rate of committing fouls-, less propensity to aerial duels.

![png](/Model/Players/dfcc1.PNG)

- C2: Mastery of the aerial game, more defensive disputes, greater ability to prevail individually, less participation with the ball.

![png](/Model/Players/dfcc2.PNG)

- C3: Less propensity to duels and tackles, fewer fouls, greater participation with the ball.

![png](/Model/Players/dfcc3.PNG)

**Full-Backs**

For this position, the most explanatory variables and those that contribute to a greater extent to compose the differences between players and group them are the following.

![png](/Model/Players/lat_seg.png)

- C1: Playmaker from the side, less defensive intensity but does not gain height in attack. A lot of participation in buildup and elaboration.

![png](/Model/Players/latc1.PNG)

- C2: Inverted full-back, joining the attack towards interior areas.

![png](/Model/Players/latc2.PNG)

- C3: Gains depth and seek for crosses, high involvement in the generation of chances and less participation in buildup.

![png](/Model/Players/latc3.PNG)

- C4: Rather classic profile. Lower rate of crosses, greater volume of defensive activity, little arrival in areas of the last pass.

![png](/Model/Players/latc4.PNG)

**Midfielders**

For this position, the most explanatory variables and those that contribute to a greater extent to compose the differences between players and segment them are the following.

![png](/Model/Players/cen_seg.png)

- C1: More offensive profile, fewer defensive actions, greater participation in creating chances, drop to the wing and receptions close to the area.

![png](/Model/Players/cenc1.PNG)

- C2: Pivot, high volume of defensive tasks, little participation in the final third, high rate of disputes and receptions at the base of the play.

![png](/Model/Players/cenc2.PNG)

- C3: Ambivalent profile. Covers a lot of area, with and without the ball. Itinerant organizers in teams that appear to build at different heights and midfielders with a high ability to reach the area to finish off.

![png](/Model/Players/cenc3.PNG)


**Attacking Midfielders / Wingers**

For this position, the most explanatory variables and those that contribute to a greater extent to compose the differences between players and segment them are the following.

![png](/Model/Players/mco_seg.png)

- C1: Winger on natural foot, facer and dribbler. High rate of crosses in the baseline. Wingers that play very high are included.

![png](/Model/Players/mcoc1.PNG)

- C2: Self-sufficient players, always in the center or on the wing with a different leg. High incidence in the area (high values ​​of the last pass and own opportunities enjoyed).

![png](/Model/Players/mcoc2.PNG)

- C3: Inverted wingers or attacking midfielders with high value generated through the pass and a lot of participation in construction. Some, accustomed to playing in the position of 10 or as interiors in 4-3-3.

![png](/Model/Players/mcoc3.PNG)

**Forwards**

For this position, the most explanatory variables and those that contribute to a greater extent to compose the differences between players and segment them are the following.

![png](/Model/Players/del_seg.png)

- C1: Penalty-area-based striker, dribbler in short spaces, less movement without the ball, less participation in possession.

![png](/Model/Players/dcc1.PNG)

- C2: Greater influence in the generation of opportunities, both for themselves and for their teammates. Striker with a wide range of movements and self-sufficient.

![png](/Model/Players/dcc2.PNG)

- C3: More aerial duels and dominance of the game by imposing his rule up high. Static but with participation in the game -receiving from behind, winning duels that lead to second play-.

![png](/Model/Players/dcc3.PNG)

- C4: More defensive effort and participation in development, second striker / false nine profile. He moves and stays wide, dribbles and generates supporting plays for surrounding players.

![png](/Model/Players/dcc4.PNG)


### Distance Algorithms

The two processes outlined below aim to turn all of the above analytical modeling into actionable knowledge. Both have in common, on the one hand, their purpose of serving the sports management in the task of refining and basing the player recruitment process on data; and, on the other, the use of the Euclidean distance as a means to determine the conclusions. However, the starting point and the meanings are very different, which differentiates them and also complements them, as we will see in the practical cases.

- The **player_Similarities** function measures the degree of similarity between a player and the rest of his homonyms. It takes a player and returns a table with the Euclidean distance with respect to the most similar ones. It should be noted that, although there may be a relationship between the results of this model and the level of the player, what it tries to illustrate is exclusively the similarity between one player and another, leaving aside their level and quality.


- The **team_Mapping** function takes some base data corresponding to a specific game model and returns the players whose attributes are closest -and, therefore, are most appropriate to adapt to that context-. It takes a subposition and a team, and returns a table with the Euclidean distance from the closest players to that base data.

The functions are collected in the script: *clustering_functions.py*.

*Player_Similarities* fits the distance measurement function prototype, and its results are easy to see on any advanced player analytics website. This function becomes relevant in practice when we are interested in replacing a footballer whose profile and role is very defined. A clear case will happen when a purely selling club has to release a key piece and wants to look for a replacement that is as similar as possible, to avoid redefining its game model.

![png](/Model/Players/rodri1.PNG)

![png](/Model/Players/rodri2.png)

The figures shown above show the results of the model in the case of the players most similar to Rodrigo Hernández, the Manchester City midfielder. Said process, as we have already indicated, does not assess the adequacy of the player to Rodri's ecosystem, but simply assesses similarities between the characteristics of the players in his position. Among the names that are returned is, obviously, that of Sergio Busquets, his predecessor in the national team and pivot of Barcelona, ​​which in turn is one of the teams whose model most closely resembles that of the team currently led by Pep Guardiola. However, there are also players who act in contexts other than the model of possession and position play -John Lundstram (Rangers) or Thiago Alcántara (Liverpool)-.

As a practice-oriented example, we can apply the model to the context of Sevilla FC, which in the summer had to deal with the departures of its starting central defenders, Diego Carlos and Jules Koundé. We take a look at the role returns for the French centre-back below, looking at the players who most resemble him and a comparison with some of the biggest names.

![png](/Model/Players/kounde2.PNG)

![png](/Model/Players/kounde1.png)

We endorse the results of the model by comparing Jules Koundé and Samy Mmaee, the Moroccan central defender for Ferencvaros. We found that the morphology of the radar, indeed, shows great similarity, and that, for general purposes, both players share strengths and weaknesses despite the level differences that may exist.

![png](/Model/Players/mmaee.PNG)

In fact, we can see that, despite the divergence in the market value -€60M vs. €1M-, the French center-back does not show a clear superiority -with the exception of the recoveries made in the rival field-, which may imply a certain competitive advantage for the Nervionenses, since they could eventually find a financial replacement in the Ferencvaros center-back very appetizing, and that, given the figure for which it was sold to Koundé, it would be an economic operation of great benefit while, sportingly, damage would be limited. The results are also encouraging if we measure Koundé with Castello Lukeba, the Lyon centre-back who, obviously, has a superior lineup to Mmaee and a value that, for economic purposes, would probably keep him off Sevilla's radar.

![png](/Model/Players/kounde3.png)

**Team_Mapping** groups all the teams belonging to the same series of team clusters and extracts an average value of all their metrics. That will be the point of reference, for the teams that belong to said clusters, when looking for similar players.

The fact that the model returns players that are close to an average portrait implies that there will be similarities at the profile level, but obviously it does not mean that the level of the player in question corresponds to that of the team in question. For this reason, and also because the economic reality of the club itself must be taken into account, it is important to take into account the variables of market value, age or the most cutting-edge performance metrics. In the tables below we will show the Instat index as a measuring element of the player's level, and can act as a filter.

It is worth to focus on the similarities, and for this we will show some quick examples, going through the team and the need in question, and in which we will be returned a list with the footballers closest to the metrics of the game model of the indicated team.

![png](/Model/Players/city_centrales.PNG)

We see in the example the list of central defenders that are closest to the metrics of the Manchester City model for that position, establishing a filter of *InStat Index* that is not very exclusive. As we saw earlier in the profiling, the center backs of this cluster are characterized by excellent footwork and extraordinary precision when changing direction. They must also be quick, as Pep Guardiola's team places the defensive line very far from their goalkeeper, and with the ability to anticipate.

We take Manuel Akanji as an example below, and we measure his similarity on the radar with one of the central citizens, the Spanish international Aymeric Laporte. We also add the dashboard of what the model considers to be the center-back that best fits in this environment: Nino, a promising Brazilian defender for Fernando Diniz's buoyant Fluminense -who, in fact, puts his two center-backs on that table-.

![png](/Model/Players/akanji.png)

![png](/Model/Players/nino.PNG)

To check the ambivalence of the model, we will now face another problem in defense, but this time we will look for left-backs adapted to the Chelsea style, a team that, although it defends up front and also demands a clean start from behind, has a fundamental character that must be taken into account in the model: he has been in a line of 3 center-backs for years, with which he must find players who adapt to the role of winger. Once again, full-backs are shown who pass the same permissive InStat Index filter -which simply discards players who are clearly not comparable, in terms of competition level, to the Chelsea context- but with an age of no more than 30 years.

![png](/Model/Players/inter.PNG)

Observing the table, we find that the model returns very offensive profiles -clusters 1 (band organizer with less defensive activity) and 3 (deep winger, crosser and chance generator who participates less in construction). Within the diversity, we can verify that there are names that either have a great facility to be differential in higher areas of the field -Alex Telles, Yuri Berchiche- or base their game on an excellent ability to cross -Lucas Digne- or have Experience forming as wingers in defenses of three -Reguilón, Cucurella, Renan Lodi, Fortuna, Doumbia, Estupiñán or Caio Henrique-. There are even players who, in their past or occasionally in the present, act/acted in the extreme position, in the case of the aforementioned Cucurella or Johan Mojica.

![png](/Model/Players/estupinan_alonso.png)

On a practical level, a sports management that places data at the center of decision-making cannot limit itself to running models that compare players, decontextualizing them from the environment in which they develop their game, while not always resorting to the market to look for a profile that is as similar as possible to a player who has left the club -there are many other different contexts that motivate a club to make a transfer-. For these reasons, the optimal strategy is to include the game model as the main axis to profile the player, and establish the metrics that determine the way of playing as the axis to measure the adequacy of a player to the context of the club that intends to fill a vacancy. . Through **team_mapping** we can find out, when faced with a need that arises in the squad or a possible improvement, which are the players in the position in question that are closest to the game model and the tactical conditions we have. This process can provide important information regarding the profile necessary to incorporate, and can help narrow down potential candidates.

If, in addition, said need that makes the club go to the transfer market is motivated by the departure of an important footballer, it will be convenient to combine the result of this function with that of the previous one, and evaluate in detail from the technical perspective -look, watch matches, find out about his personality and environment, assess his injury history, assess whether negotiation with the club to which he belongs is possible, etc- the conditions of those players who appear on the resulting lists and fit into the financial policy of the club .

If the incorporation that is intended to be carried out really satisfies a role that did not exist until today in the template, it would not make so much sense to use the **player_similarities** function, while **team_mapping**, used correctly, would continue to offer some relevant information.

### The App Explained

We begin the last appendix of the project by analyzing the script that generates the different elements available in the application.

#### Application coding

The execution and deployment of the distance functions of the model and of all the elements that are part of the front of the page is done in the *streamlit_imp.py* script. However, the matplotlib and seaborn bar and radar plots that can be distinguished in the app are formulated in the *players_plotting.py* script.



#### Deployment and Documentation

The app is uploaded to the Streamlit Cloud via a [Github repository](https://github.com/alexfrf/scouting-profiler), specifying a number of basic layout parameters within *config.toml* in the *.streamlit* folder. The aforementioned repository obviously houses the source data and all the necessary code to be able to create the model and the app.

#### Displayable Elements

The app is divided into five very different sections:

- **Vertical Drop-down Bar with Filter Widgets**: it is located on the left of the screen and contains the modifiable parameters that affect the model results. There are three filters that directly affect the model, since it will return the desired number of most suitable players based on the position (vacant) and the team. From there, the results can be shaped according to a series of variables:

   - Market value (transfermarkt) limit
   - Age limit
   - Good Leg
   - INSTAT Quality Index
   - Competitions: allows you to choose or omit championships in the model. This will return players from the chosen leagues
   - Teams: Allows you to choose or omit teams in the model. This will return players from the chosen clubs
   
 

- **Title and Introduction**: it is located in the header and tries to explain the motivation of the application and its operation, detailing the origin of the data.



- **Model Return**: shows the X players best suited to the selected position and team, in a table detailing the player's club, age, country, market value, Instat index, degree of similarity with the game model and cluster (group) to which the player is assigned. On the right, it shows the most determining variables when establishing clustering and the distances between players.

![png](/Model/Players/streamlit.PNG)

- **Dashboard**: it is located in the part that, vertically, we could call "central" in the application. This section contains the key information of the selected player, either one of those listed in the returns table or any other that we want to consult and compare. Four views are available:
    - Simple radar of the selected player (upper widget, by default it is the 1st of the returns table). Compare the player (purple) with the average number of players in his position in his league (red) and the maximum for each variable (light grey).
    - On the right, the percentiles are shown in a bar graph. In this case, the player is compared with all the players in his basic position (if he is a left winger, he will be compared with all the midfielders and wingers) from all the leagues selected in the filter section.
    - Below is a comparative radar that, by default, measures the selected player with the footballer who occupies the chosen position in the club that we have chosen in the filter section. Likewise, any comparison can be made using the widgets that appear just above the radar.
    - The fourth quadrant shows the most similar players -without taking into account the game model of the teams- to the football player selected in the first widget-. We can also see the cluster of each of them and their degree of similarity.

![png](/Model/Players/radar1.PNG)

![png](/Model/Players/radar2.png)

- **Explanation of Clusters**: this last section explains the segmentation carried out by the model, both for team game models (left), differentiating by game phase, and for players (right), showing the division for position selected. For each group, both in players and in teams, explain the basic characteristics of each cluster, adding, in tables, the members of each group (use widgets). In addition, in the case of teams, a KPI is shown with the average number of expected points for each cluster.

![png](/Model/Players/dash_clusters.PNG)
