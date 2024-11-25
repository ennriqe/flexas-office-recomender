import os
# Import TensorFlow first and configure it
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Now continue with other imports
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import folium
from streamlit_folium import st_folium

def load_dependencies():
    """Load the saved model and scaler"""
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    product_cols = joblib.load('product_cols.pkl')
    return model, scaler, product_cols

def create_input_interface():
    st.title("Office Building Recommendation System")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        # Hardcoded values (hidden from interface)
        n_buildings = 5  # Hardcoded
        time_between = 9001  # Hardcoded
        
        max_workstations = st.number_input(
            "Maximum Workstations",
            min_value=0,
            max_value=1000,
            value=20
        )
        
        min_workstations = st.number_input(
            "Minimum Workstations",
            min_value=0,
            max_value=1000,
            value=10
        )
        
        estimated_value = st.number_input(
            "Estimated Value",
            min_value=0,
            max_value=1000000,
            value=5000,
            # Prevent empty values
            step=1
        )

    with col2:
        st.subheader("Location Preferences")
        
        # City selection with search
        cities = [
            'Amsterdam', 'Paris', 'Rotterdam', 'Utrecht', 'Den Haag',
            'Eindhoven', 'Almere', 'Berlin', 'Amsterdam Zuidoost',
            'Haarlem', 'Breda', 'Den Bosch', 'Amersfoort', 'München',
            'Arnhem', 'Hoofddorp', 'Bruxelles', 'Brussels', 'Hamburg',
            'Amstelveen', 'Antwerpen', 'Rijswijk', 'Brussel', 'Groningen',
            'Leiden', 'Zwolle', 'Capelle aan den IJssel', 'Schiphol',
            'Nieuwegein', 'Frankfurt', 'Nijmegen', 'Gent', 'Zoetermeer',
            'Maastricht', 'Delft', 'Alkmaar', 'Zaventem', 'Hilversum',
            'Apeldoorn', 'Düsseldorf', 'Tilburg', 'Vianen', 'Antwerp',
            'Diemen', 'Gouda', 'Munich', 'Barendrecht', 'Dordrecht',
            'Schiedam', 'Mechelen', 'Luxembourg', 'Deventer', 'Veenendaal',
            'Köln', 'Diegem', 'Houten', 'Heerlen', 'Aalsmeer', 'Lelystad',
            'Ede', 'London', 'Neuilly-sur-Seine', 'Zaandam',
            'Boulogne-Billancourt', 'Leuven', 'Venlo', 'Woerden',
            'Schiphol-Rijk', 'Bussum', 'Roosendaal'
        ]
        
        selected_city = st.selectbox(
            "Primary City (type to search)",
            options=sorted(cities),  # Sort cities alphabetically for easier searching
            help="Start typing to search for a city"
        )
        

        
        # Create three columns for better layout of checkboxes
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            conventional = st.selectbox(
                "Conventional Space",
                options=[None, True, False],
                format_func=lambda x: "No preference" if x is None else ("Yes" if x else "No")
            )
            
            managed = st.selectbox(
                "Managed Office",
                options=[None, True, False],
                format_func=lambda x: "No preference" if x is None else ("Yes" if x else "No")
            )
            
        with filter_col2:
            serviced = st.selectbox(
                "Serviced Office",
                options=[None, True, False],
                format_func=lambda x: "No preference" if x is None else ("Yes" if x else "No")
            )
            
            furnished = st.selectbox(
                "Fully Furnished",
                options=[None, True, False],
                format_func=lambda x: "No preference" if x is None else ("Yes" if x else "No")
            )
        
        # Hardcoded year (hidden from interface)
        selected_year = 2024

    return {
        'n_buildings_offered': n_buildings,  # Hardcoded to 5
        'new_maximumworkstation': max_workstations,
        'new_minimumworkstations': min_workstations,
        'time_between_viewing_and_opportunity': time_between,  # Hardcoded to 9001
        'estimatedvalue': estimated_value if estimated_value is not None else 0,  # Handle None values
        'selected_city': selected_city,
        'selected_year': selected_year,  # Hardcoded to 2024
        'conventional': conventional,
        'managed': managed,
        'serviced': serviced,
        'furnished': furnished
    }

def prepare_model_input(user_input, scaler):
    """Convert user input into model-ready format"""
    # Define columns in exact order as expected by scaler
    columns = [
        'n_buildings_offered', 'new_maximumworkstation',
        'new_minimumworkstations', 'time_between_viewing_and_opportunity',
        'estimatedvalue', 'Visited', 'year_2015', 'year_2016', 'year_2017',
        'year_2018', 'year_2019', 'year_2020', 'year_2021', 'year_2022',
        'year_2023', 'year_2024', 'city_Amsterdam', 'city_Paris',
        'city_Rotterdam', 'city_Utrecht', 'city_Den Haag', 'city_Eindhoven',
        'city_Almere', 'city_Berlin', 'city_Amsterdam Zuidoost', 'city_Haarlem',
        'city_Breda', 'city_Den Bosch', 'city_Amersfoort', 'city_München',
        'city_Arnhem', 'city_Hoofddorp', 'city_Bruxelles', 'city_Brussels',
        'city_Hamburg', 'city_Amstelveen', 'city_Antwerpen', 'city_Rijswijk',
        'city_Brussel', 'city_Groningen', 'city_Leiden', 'city_Zwolle',
        'city_Capelle aan den IJssel', 'city_Schiphol', 'city_Nieuwegein',
        'city_Frankfurt', 'city_Nijmegen', 'city_Gent', 'city_Zoetermeer',
        'city_Maastricht', 'city_Delft', 'city_Alkmaar', 'city_Zaventem',
        'city_Hilversum', 'city_Apeldoorn', 'city_Düsseldorf', 'city_Tilburg',
        'city_Vianen', 'city_Antwerp', 'city_Diemen', 'city_Gouda',
        'city_Munich', 'city_Barendrecht', 'city_Dordrecht', 'city_Schiedam',
        'city_Mechelen', 'city_Luxembourg', 'city_Deventer', 'city_Veenendaal',
        'city_Köln', 'city_Diegem', 'city_Houten', 'city_Heerlen',
        'city_Aalsmeer', 'city_Lelystad', 'city_Ede', 'city_London',
        'city_Neuilly-sur-Seine', 'city_Zaandam', 'city_Boulogne-Billancourt',
        'city_Leuven', 'city_Venlo', 'city_Woerden', 'city_Schiphol-Rijk',
        'city_Bussum', 'city_Roosendaal', 'city_Unknown',
        'estimatedvalue_missing'
    ]
    
    # Create DataFrame with all columns initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    
    # Fill in the user input values
    input_df.loc[0, 'n_buildings_offered'] = user_input['n_buildings_offered']
    input_df.loc[0, 'new_maximumworkstation'] = user_input['new_maximumworkstation']
    input_df.loc[0, 'new_minimumworkstations'] = user_input['new_minimumworkstations']
    input_df.loc[0, 'time_between_viewing_and_opportunity'] = user_input['time_between_viewing_and_opportunity']
    input_df.loc[0, 'estimatedvalue'] = user_input['estimatedvalue']
    input_df.loc[0, 'Visited'] = 1
    input_df.loc[0, f"year_{user_input['selected_year']}"] = 1
    input_df.loc[0, f"city_{user_input['selected_city']}"] = 1
    
    # Handle missing values
    input_df.loc[0, 'estimatedvalue_missing'] = 1 if pd.isna(user_input['estimatedvalue']) else 0
    print(input_df.to_dict())
    # Scale the input
    input_scaled = scaler.transform(input_df)

    # print(pd.DataFrame(input_scaled).to_dict())
    
    return input_scaled

def display_recommendations(predictions, buildings_df, product_cols, filters):
    st.subheader("Top Building Recommendations")
    
    try:
        # Convert coordinates to float when loading the dictionary
        buildings_df['fl_drupallongitude'] = pd.to_numeric(buildings_df['fl_drupallongitude'], errors='coerce')
        buildings_df['fl_drupallatitude'] = pd.to_numeric(buildings_df['fl_drupallatitude'], errors='coerce')
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'building_id': product_cols,
            'probability': predictions[0]
        })
        
        # Add productid column for merging
        recommendations['productid'] = recommendations['building_id'].str.replace('prod_', '')
        
        # Merge with buildings_df to get statecode and other info
        recommendations = recommendations.merge(
            buildings_df[[
                'productid', 'name', 'fl_drupallongitude', 'fl_drupallatitude', 
                'statecode', 'fl_drupaltypeemptyspaceconventional', 
                'fl_drupaltypemanagedoffice', 'fl_drupaltypeservicedoffice',
                'fl_drupalservicesfullyfurnished', 'new_region'
            ]], 
            on='productid', 
            how='left'
        )
        
        # Start with base filters
        mask = (
            (recommendations['statecode'] == 0) & 
            (~recommendations['name'].str.contains(" - Address ", na=False)) &
            (recommendations['new_region'] == filters['selected_city'])  # Match new_region with selected city
        )
        
        # Add type filters if specified
        if filters['conventional'] is not None:
            mask &= (recommendations['fl_drupaltypeemptyspaceconventional'] == filters['conventional'])
        if filters['managed'] is not None:
            mask &= (recommendations['fl_drupaltypemanagedoffice'] == filters['managed'])
        if filters['serviced'] is not None:
            mask &= (recommendations['fl_drupaltypeservicedoffice'] == filters['serviced'])
        if filters['furnished'] is not None:
            mask &= (recommendations['fl_drupalservicesfullyfurnished'] == filters['furnished'])
            
        # Apply all filters
        recommendations = recommendations[mask]
        
        # Sort by probability and get top 10
        recommendations = recommendations.sort_values('probability', ascending=False)
        top_10 = recommendations.head(10)
        top_10 = top_10.reset_index(drop=True)

        # Create two columns for the layout
        col_list, col_map = st.columns([1, 1])
        
        with col_list:
            st.write("Top 10 Recommended Buildings (Active Only):")
            for idx, row in top_10.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{row['name']}**")
                with col2:
                    st.write(f"Confidence: {row['probability']:.2%}")
                with col3:
                    has_coords = not (pd.isna(row['fl_drupallatitude']) or pd.isna(row['fl_drupallongitude']))
                    st.write("📍" if has_coords else "❌")
                st.divider()
        
        with col_map:
            # Prepare data for the map
            map_data = top_10[['fl_drupallatitude', 'fl_drupallongitude', 'name']].copy()
            map_data = map_data.rename(columns={
                'fl_drupallatitude': 'latitude',
                'fl_drupallongitude': 'longitude',
                'name': 'building_name'
            })
            map_data = map_data.dropna()
            
            if not map_data.empty:
                st.write(f"Showing {len(map_data)} buildings with valid coordinates")
                
                # Create a folium map centered on the mean coordinates
                m = folium.Map(
                    location=[map_data['latitude'].mean(), map_data['longitude'].mean()],
                    zoom_start=8
                )
                
                # Add markers with labels
                for idx, row in map_data.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=row['building_name'],  # Only show building name in popup
                        tooltip=f"#{idx + 1}",  # Keep the index in hover tooltip
                        icon=folium.Icon(
                            icon='building',
                            prefix='fa',
                            color='red'
                        )
                    ).add_to(m)
                
                # Display the map
                st_folium(m, width=400, height=400)
            else:
                st.warning("No coordinate data available for mapping")
            
    except Exception as e:
        st.error(f"Error creating recommendations: {str(e)}")
        st.write("Debug info:")
        st.write("Predictions shape:", predictions.shape)
        st.write("Product cols length:", len(product_cols))

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "flexas2024":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def main():
    if not check_password():
        st.stop()  # Do not continue if check_password is False
        
    try:
        # Initialize session state for recommendations if it doesn't exist
        if 'recommendations_made' not in st.session_state:
            st.session_state.recommendations_made = False
            
        model, scaler, product_cols = load_dependencies()
        buildings_df = pd.read_csv('buildings_df.csv')
        
        # Create interface and get user input
        user_input = create_input_interface()
        
        # Add a submit button
        if st.button("Get Recommendations"):
            # Prepare input for model
            model_input = prepare_model_input(user_input, scaler)
            
            # Get predictions
            predictions = model.predict(model_input)
            
            # Store predictions and set flag
            st.session_state.predictions = predictions
            st.session_state.recommendations_made = True
        
        # Display recommendations if they exist
        if st.session_state.recommendations_made:
            display_recommendations(st.session_state.predictions, buildings_df, product_cols, user_input)
            
    except Exception as e:
        st.error(f"Error running the app: {str(e)}")

if __name__ == "__main__":
    main()
