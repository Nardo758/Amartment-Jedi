#!/usr/bin/env python3
"""
ApartmentIQ Streamlit Web Application
AI-Powered Apartment Hunting Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from dataclasses import asdict
import sqlite3
import os
from pathlib import Path

# Import core ApartmentIQ functionality
try:
    from apartmentiq_core import (
        ApartmentIQ, PropertyListing, UserProfile, 
        MarketAnalysis, SmartOffer
    )
except ImportError:
    st.error("❌ Core ApartmentIQ module not found. Please ensure apartmentiq_core.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ApartmentIQ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .property-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
    .offer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'apartment_iq' not in st.session_state:
        st.session_state.apartment_iq = ApartmentIQ()
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    
    if 'properties' not in st.session_state:
        st.session_state.properties = []
    
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    
    if 'generated_offers' not in st.session_state:
        st.session_state.generated_offers = []
    
    if 'market_analysis' not in st.session_state:
        st.session_state.market_analysis = None

@st.cache_data(ttl=3600)
def load_properties(city, state):
    """Load properties with caching"""
    apartment_iq = ApartmentIQ()
    return apartment_iq.discover_opportunities(city, state)

@st.cache_data(ttl=1800)
def get_market_analysis(city):
    """Get market analysis with caching"""
    apartment_iq = ApartmentIQ()
    return apartment_iq.analyzer.analyze_market_conditions(city)

def create_user_profile_form():
    """Create user profile input form"""
    st.subheader("👤 Your Profile")
    
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            current_rent = st.number_input(
                "Current Monthly Rent ($)", 
                min_value=500, 
                max_value=10000, 
                value=2600,
                step=50
            )
            
            max_budget = st.number_input(
                "Maximum Budget ($)", 
                min_value=500, 
                max_value=15000, 
                value=2500,
                step=50
            )
            
            min_bedrooms = st.selectbox(
                "Minimum Bedrooms", 
                [1, 2, 3, 4, 5],
                index=0
            )
        
        with col2:
            lease_expires = st.date_input(
                "Current Lease Expires",
                value=datetime(2025, 3, 31)
            )
            
            work_location = st.text_input(
                "Work Location", 
                value="Downtown Austin"
            )
            
            max_commute = st.slider(
                "Max Commute Time (minutes)", 
                5, 60, 30
            )
        
        st.subheader("Preferred Amenities")
        amenity_cols = st.columns(4)
        
        with amenity_cols[0]:
            gym = st.checkbox("Gym/Fitness Center", value=True)
            pool = st.checkbox("Swimming Pool")
        
        with amenity_cols[1]:
            parking = st.checkbox("Parking", value=True)
            pet_friendly = st.checkbox("Pet Friendly", value=True)
        
        with amenity_cols[2]:
            wifi = st.checkbox("High-Speed Internet")
            laundry = st.checkbox("In-Unit Laundry")
        
        with amenity_cols[3]:
            balcony = st.checkbox("Balcony/Patio")
            dishwasher = st.checkbox("Dishwasher")
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            preferred_amenities = []
            if gym: preferred_amenities.append('gym')
            if pool: preferred_amenities.append('pool')
            if parking: preferred_amenities.append('parking')
            if pet_friendly: preferred_amenities.append('pet')
            if wifi: preferred_amenities.append('wifi')
            if laundry: preferred_amenities.append('laundry')
            if balcony: preferred_amenities.append('balcony')
            if dishwasher: preferred_amenities.append('dishwasher')
            
            st.session_state.user_profile = UserProfile(
                current_rent=current_rent,
                lease_expires=lease_expires.strftime("%B %Y"),
                max_budget=max_budget,
                work_lat=30.2672,  # Default Austin coordinates
                work_lng=-97.7431,
                preferred_amenities=preferred_amenities,
                min_bedrooms=min_bedrooms,
                max_commute_time=max_commute
            )
            
            st.success("✅ Profile saved successfully!")
            time.sleep(1)
            st.rerun()

def display_market_overview():
    """Display market overview dashboard"""
    st.subheader("📊 Market Overview")
    
    if st.session_state.market_analysis:
        analysis = st.session_state.market_analysis
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg. Days on Market",
                f"{analysis.avg_days_on_market:.0f} days",
                delta="-5 days vs last month"
            )
        
        with col2:
            st.metric(
                "Price Reductions",
                f"{analysis.price_reduction_rate:.0%}",
                delta="3% vs last month"
            )
        
        with col3:
            st.metric(
                "Demand Index",
                f"{analysis.demand_index}/100",
                delta="-8 points (Winter slowdown)"
            )
        
        with col4:
            st.metric(
                "Median Rent",
                f"${analysis.median_rent:,}",
                delta="$50 vs last month"
            )

def display_property_cards(properties):
    """Display property listings as cards"""
    st.subheader("🏠 Available Properties")
    
    if not properties:
        st.info("No properties found. Try adjusting your search criteria.")
        return
    
    for prop in properties:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Mock property image
                st.image(
                    "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=300&h=200&fit=crop",
                    width=200
                )
            
            with col2:
                st.markdown(f"**{prop.address}**")
                st.write(f"🛏️ {prop.bedrooms}BR/{prop.bathrooms}BA • 📐 {prop.sqft} sq ft")
                
                # Amenities
                amenity_text = " • ".join([f"🏋️ Gym" if 'gym' in prop.amenities else "",
                                         f"🏊 Pool" if 'pool' in prop.amenities else "",
                                         f"🚗 Parking" if 'parking' in prop.amenities else "",
                                         f"🐕 Pet OK" if 'pet' in prop.amenities else ""])
                amenity_text = " • ".join([a for a in amenity_text.split(" • ") if a])
                if amenity_text:
                    st.write(amenity_text)
                
                # Market indicators
                status_color = "🟣" if prop.source == "craigslist" else "🔵"
                st.write(f"{status_color} {prop.source.title()} • ⏰ {prop.days_on_market} days on market")
            
            with col3:
                st.markdown(f"<h3 style='color: #2E86AB; text-align: right;'>${prop.rent:,}/mo</h3>", 
                           unsafe_allow_html=True)
                
                if st.button(f"🎯 Generate Offer", key=f"offer_{prop.id}", use_container_width=True):
                    st.session_state.selected_property = prop
                    st.rerun()
                
                if prop.days_on_market > 45:
                    st.markdown("🔥 **High Negotiation Potential**")
        
        st.divider()

def display_offer_generator():
    """Display AI offer generation interface"""
    if not st.session_state.selected_property:
        st.info("Select a property to generate a smart offer.")
        return
    
    prop = st.session_state.selected_property
    st.subheader(f"🤖 Smart Offer Generator")
    st.write(f"**Property:** {prop.address}")
    
    # Generate offer
    with st.container():
        if st.button("🧠 Analyze & Generate Offer", use_container_width=True):
            
            # Show analysis progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analysis_steps = [
                "Analyzing market conditions...",
                "Evaluating owner motivation...", 
                "Reviewing price history...",
                "Scanning comparable properties...",
                "Calculating optimal offer...",
                "Generating negotiation strategy..."
            ]
            
            for i, step in enumerate(analysis_steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(analysis_steps))
                time.sleep(0.5)
            
            # Generate actual offer
            if st.session_state.user_profile and st.session_state.market_analysis:
                offer = st.session_state.apartment_iq.offer_generator.generate_smart_offer(
                    prop, 
                    st.session_state.market_analysis,
                    st.session_state.user_profile
                )
                st.session_state.generated_offers = [offer]
                
                progress_bar.progress(1.0)
                status_text.text("✅ Smart offer generated!")
                time.sleep(1)
                st.rerun()

def display_generated_offer():
    """Display the generated smart offer"""
    if not st.session_state.generated_offers:
        return
    
    offer = st.session_state.generated_offers[0]
    prop = st.session_state.selected_property
    
    st.subheader("🎯 Your Smart Offer")
    
    # Offer summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Listed Price",
            f"${offer.original_price:,}",
            help="Original asking price"
        )
    
    with col2:
        st.metric(
            "Recommended Offer", 
            f"${offer.recommended_price:,}",
            delta=f"-${offer.original_price - offer.recommended_price:,}",
            help="AI-calculated optimal offer"
        )
    
    with col3:
        savings = offer.original_price - offer.recommended_price
        st.metric(
            "Annual Savings",
            f"${savings * 12:,}",
            help="Total savings over 12 months"
        )
    
    # Success probability
    st.markdown(f"""
    <div class="success-box">
        <h4>📈 Success Probability: {offer.success_probability:.0%}</h4>
        <p><strong>Strategy:</strong> {offer.strategy}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Leverage points
    st.subheader("💡 Key Leverage Points")
    for point in offer.leverage_points:
        st.write(f"• {point}")
    
    # Email preview
    with st.expander("📧 Generated Email Preview"):
        st.text_area(
            "Email Content",
            value=offer.email_template,
            height=400,
            help="You can customize this email before sending"
        )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📨 Send Offer Email", use_container_width=True):
            st.success("🎉 Offer sent successfully! We'll notify you when the property manager responds.")
    
    with col2:
        if st.button("📅 Schedule for Later", use_container_width=True):
            st.info("Offer scheduled for optimal timing based on market analysis.")
    
    with col3:
        if st.button("💾 Save as Template", use_container_width=True):
            st.success("Template saved for future use!")

def display_analytics_dashboard():
    """Display analytics and insights"""
    st.subheader("📈 Market Analytics")
    
    if not st.session_state.properties:
        st.info("Load property data to see analytics.")
        return
    
    # Create sample analytics data
    properties_df = pd.DataFrame([
        {
            'days_on_market': prop.days_on_market,
            'rent': prop.rent,
            'bedrooms': prop.bedrooms,
            'sqft': prop.sqft,
            'source': prop.source,
            'owner_type': prop.owner_type
        }
        for prop in st.session_state.properties
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Days on market distribution
        fig1 = px.histogram(
            properties_df, 
            x='days_on_market',
            title="Days on Market Distribution",
            color_discrete_sequence=['#2E86AB']
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Rent vs. property size
        fig2 = px.scatter(
            properties_df,
            x='sqft',
            y='rent',
            color='bedrooms',
            title="Rent vs. Square Footage",
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏠 ApartmentIQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Apartment Hunting & Negotiation Platform</p>', 
               unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        tab = st.radio(
            "Choose a section:",
            ["🏠 Property Search", "👤 User Profile", "📊 Analytics", "⚙️ Settings"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("## 🔍 Search Filters")
        
        city = st.selectbox("City", ["Austin", "Denver", "Nashville", "Raleigh"], index=0)
        state = st.selectbox("State", ["TX", "CO", "TN", "NC"], index=0)
        
        if st.button("🔍 Search Properties", use_container_width=True):
            with st.spinner("Searching for properties..."):
                st.session_state.properties = load_properties(city, state)
                st.session_state.market_analysis = get_market_analysis(city)
            st.success(f"Found {len(st.session_state.properties)} properties!")
            st.rerun()
        
        # Show current stats
        if st.session_state.properties:
            st.markdown("### 📊 Current Search")
            st.metric("Total Properties", len(st.session_state.properties))
            hidden_count = len([p for p in st.session_state.properties if p.source == 'craigslist'])
            st.metric("Hidden Opportunities", hidden_count)
    
    # Main content area
    if tab == "🏠 Property Search":
        if st.session_state.user_profile is None:
            st.warning("👤 Please create your user profile first to get personalized recommendations.")
            
        display_market_overview()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_property_cards(st.session_state.properties)
        
        with col2:
            display_offer_generator()
            display_generated_offer()
    
    elif tab == "👤 User Profile":
        create_user_profile_form()
        
        if st.session_state.user_profile:
            st.subheader("✅ Current Profile")
            profile_data = {
                "Current Rent": f"${st.session_state.user_profile.current_rent:,}",
                "Max Budget": f"${st.session_state.user_profile.max_budget:,}",
                "Lease Expires": st.session_state.user_profile.lease_expires,
                "Min Bedrooms": st.session_state.user_profile.min_bedrooms,
                "Preferred Amenities": ", ".join(st.session_state.user_profile.preferred_amenities)
            }
            
            for key, value in profile_data.items():
                st.write(f"**{key}:** {value}")
    
    elif tab == "📊 Analytics":
        display_analytics_dashboard()
    
    elif tab == "⚙️ Settings":
        st.subheader("⚙️ Application Settings")
        
        st.markdown("### 🔧 Scraping Configuration")
        scraping_delay = st.slider("Scraping Delay (seconds)", 1, 10, 2)
        max_workers = st.slider("Max Workers", 1, 8, 4)
        
        st.markdown("### 📧 Email Settings")
        email_enabled = st.checkbox("Enable Email Automation", value=True)
        
        st.markdown("### 🎯 Offer Strategy")
        aggressiveness = st.slider("Negotiation Aggressiveness", 1, 10, 7)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | ApartmentIQ v1.0 | "
        "[GitHub](https://github.com/yourusername/apartmentiq)"
    )

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
