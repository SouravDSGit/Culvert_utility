import requests
import json
import geopandas as gpd
### Note this code was modified on the original code obtained from https://streamstats.usgs.gov/docs/streamstatsservices/#
# Service URLS
StreamStatsServiceURLS = {
    'watershed': 'https://streamstats.usgs.gov/streamstatsservices/watershed.geojson',
    'basinCharacteristics': 'https://prodwebb.streamstats.usgs.gov/streamstatsservices/parameters.json'
    }
NSSServiceURlS = {
    'regressionRegions': 'https://streamstats.usgs.gov/nssservices/regressionregions/bylocation',
    'scenarios': 'https://streamstats.usgs.gov/nssservices/scenarios',
    'computeFlowStats': 'https://streamstats.usgs.gov/nssservices/scenarios/estimate'
}

# Declare cookies for use of consistent StreamStats servers
global cookies 
def get_state_from_point(lat, lon, usa_states_shapefile_path, state_abbr_column="stusps"):
    """
    Finds the US state abbreviation in which the given point (lat, lon) falls.
    
    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        usa_states_shapefile_path (str): Path to a known US states shapefile (should contain a state abbreviation column)
        state_abbr_column (str): Column name in the US states shapefile for state abbreviations (default 'stusps')
        
    Returns:
        str: US state abbreviation string where the point is located, or None if no state contains the point
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Create a point geometry from the lat/lon (Imp: the lat/lon must be in "EPSG:4326" crs)
        point = Point(lon, lat)  # Note: Point takes (x, y) which is (lon, lat)
        
        # Load the US states shapefile and convert to WGS84 (EPSG:4326)
        states_gdf = gpd.read_file(f"zip://{usa_states_shapefile_path}").to_crs("EPSG:4326")
        
        # Find the state that contains the point
        containing_state = states_gdf[states_gdf.contains(point)]
        
        if containing_state.empty:
            return None
            
        if state_abbr_column not in containing_state.columns:
            print(f"Column '{state_abbr_column}' not found in the states shapefile. Available columns: {containing_state.columns.tolist()}")
            return None
            
        # Extract the state abbreviation from the containing state
        state_abbr = containing_state[state_abbr_column].iloc[0]
        
        return state_abbr
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def peak_flow_usgs_regression_eqn(point_shapefile_path,usa_states_shapefile_path, state_abbr_column="stusps"):
    
    # Step 0 - Formatting the inputs based on the point shapefile of culverts
    point_gdf = gpd.read_file(point_shapefile_path).to_crs("EPSG:4326")
    # Get the point geometry from the GeoDataFrame
    point = point_gdf.geometry.iloc[0]

    # Extract the coordinates
    lon = point.x  # Longitude is the x-coordinate
    lat = point.y  # Latitude is the y-coordinate
    State = get_state_from_point(lat, lon, usa_states_shapefile_path, state_abbr_column="stusps")
    
    for index, row in point_gdf.iterrows():
        xlocation = row.geometry.x
        ylocation = row.geometry.y
        # Step 1 - Get the watershed
        print('Step 1 - Get the watershed')
        watershedURLParams = {
            'rcode': State, 
            'xlocation': xlocation, 
            'ylocation': ylocation,
            'crs': 4326,
        }
        delineatedBasin = None
        while delineatedBasin is None: # Sometimes the watershed endpoint returns with no geometry - this loop is written to keep trying until a geometry is returned
            print('Delineating basin...')
            basinDelineationResponse = requests.get(url = StreamStatsServiceURLS['watershed'], params = watershedURLParams)
            if basinDelineationResponse.status_code == 200:
                cookies = basinDelineationResponse.cookies
                try:
                    delineatedBasin = json.loads(basinDelineationResponse.content.decode('utf-8'))["featurecollection"][1]["feature"]["features"][0]["geometry"] # this will be used in step 2
                    workspaceID = json.loads(basinDelineationResponse.content.decode('utf-8'))["workspaceID"] # this will be used in step 4
                except:
                    print("No geometry returned. Retrying...")
            else:
                print("Error. Retrying...")

        # Step 2 - Get the Regression Regions associated with the watershed
        print('Step 2 - Get the Regression Regions associated with the watershed')
        regressionRegionPOSTBody = delineatedBasin # from step 1

        regressionRegions = None
        while regressionRegions == None:
            regressionRegionsResponse = requests.post(url = NSSServiceURlS['regressionRegions'], json=regressionRegionPOSTBody)
            if regressionRegionsResponse.status_code == 200:
                regressionRegions = json.loads(regressionRegionsResponse.content.decode('utf-8'))
                regressionRegionCodes = [ sub['code'] for sub in regressionRegions ] # this will be used in step 3
            else:
                print("Error.")

        # Step 3 - Get the Scenarios associated with the Regression Regions
        print('Step 3 - Get the Scenarios associated with the Regression Regions')

        scenarioURLParams = {
            'regions': State, 
            'statisticgroups': 2, # this can be updated depending on which flow statistics you want to calculate (2 is associated with Peak Flow Statistics) - you can find all statistic groups available for your region at https://streamstats.usgs.gov/docs/nssservices/#/StatisticGroups/GET/StatisticGroups
            'regressionregions': (','.join(regressionRegionCodes)) # from step 2, needs to be comma separated list
        }

        parameters = None
        while parameters == None:
            scenarioResponse = requests.get(url = NSSServiceURlS['scenarios'], params=scenarioURLParams)
            if scenarioResponse.status_code == 200:
                scenarios = json.loads(scenarioResponse.content.decode('utf-8'))[0] # this will be used in step 5
                parameters = json.loads(scenarioResponse.content.decode('utf-8'))[0]["regressionRegions"][0]["parameters"]
                parameterCodes = [ sub['code'] for sub in parameters ] # this will be used in step 4
            else:
                print("Error.")

        # Step 4 - Compute the basin characteristics
        print('Step 4 - Compute the basin characteristics')
        basinCharacteristicsURLParams = {
            'rcode': State, 
            'workspaceID': workspaceID, # from step 1
            'includeparameters': (','.join(parameterCodes)) # from step 3
        }

        basinCharacteristics = None
        while basinCharacteristics is None:
            basinCharacteristicsResponse = requests.get(url = StreamStatsServiceURLS['basinCharacteristics'], params=basinCharacteristicsURLParams, cookies=cookies)
            if basinCharacteristicsResponse.status_code == 200:
                basinCharacteristics = json.loads(basinCharacteristicsResponse.content.decode('utf-8')) # this will be used in step 5
            else:
                print("Error.")

        # Step 5 - Compute Flow Statistics
        print('Step 5 - Compute Flow Statistics')
        flowStatsURLParams = {
            'regions': State
        }
        flowStatsPOSTBody = [] 
        for counter, x in enumerate(scenarios['regressionRegions'][0]['parameters']): # from step 3 & step 4
            for p in basinCharacteristics['parameters']:
                if x['code'].lower() == p['code'].lower():
                    scenarios['regressionRegions'][0]['parameters'][counter]['value'] = p['value']
        flowStatsPOSTBody.append(scenarios)

        flowStats = None
        while flowStats == None:
            flowStatsResponse = requests.post(url = NSSServiceURlS['computeFlowStats'], params = flowStatsURLParams, json=flowStatsPOSTBody)
            if flowStatsResponse.status_code == 200:
                flowStats = json.loads(flowStatsResponse.content.decode('utf-8'))
                print(flowStats)
            else:
                print("Error.")