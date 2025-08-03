#!/usr/bin/env python3
"""
ApartmentIQ: AI-Powered Vacancy & Relocation Platform - Streamlit App
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PropertyListing:
    """Data structure for apartment listings"""
    id: str
    address: str
    rent: int
    bedrooms: int
    bathrooms: float
    sqft: int
    amenities: List[str]
    days_on_market: int
    price_history: List[Dict[str, Any]]
    source: str
    listing_url: str
    property_manager: str
    owner_type: str  # 'individual', 'small_investor', 'corporate', 'reit'
    lat: float
    lng: float
    scraped_at: datetime
    
@dataclass 
class UserProfile:
    """User preferences and constraints"""
    current_rent: int
    lease_expires: str
    max_budget: int
    work_lat: float
    work_lng: float
    preferred_amenities: List[str]
    min_bedrooms: int
    max_commute_time: int

@dataclass
class MarketAnalysis:
    """Market intelligence data"""
    avg_days_on_market: float
    price_reduction_rate: float
    demand_index: int
    seasonal_factor: float
    competitive_units: int
    median_rent: int

@dataclass
class SmartOffer:
    """Generated offer details"""
    property_id: str
    recommended_price: int
    original_price: int
    strategy: str
    success_probability: float
    leverage_points: List[str]
    email_template: str
    reasoning: str

class ApartmentScraper:
    """Multi-source apartment listing scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def scrape_apartments_com(self, city: str, state: str) -> List[PropertyListing]:
        """Scrape Apartments.com for listings"""
        logger.info(f"Scraping Apartments.com for {city}, {state}")
        
        try:
            mock_listings = [
                {
                    'id': 'apt_001',
                    'address': '2847 Riverside Dr, Austin, TX',
                    'rent': 2400,
                    'bedrooms': 2,
                    'bathrooms': 2.0,
                    'sqft': 1150,
                    'amenities': ['pool', 'gym', 'parking', 'pet'],
                    'days_on_market': 67,
                    'source': 'apartments.com',
                    'property_manager': 'Riverside Properties LLC',
                    'owner_type': 'small_investor'
                },
                {
                    'id': 'apt_002', 
                    'address': '1234 Tech Ridge Blvd, Austin, TX',
                    'rent': 2800,
                    'bedrooms': 2,
                    'bathrooms': 2.0,
                    'sqft': 1200,
                    'amenities': ['pool', 'gym', 'wifi', 'parking'],
                    'days_on_market': 23,
                    'source': 'apartments.com',
                    'property_manager': 'TechRidge Management',
                    'owner_type': 'corporate'
                }
            ]
            
            listings = []
            for listing_data in mock_listings:
                listing = PropertyListing(
                    **listing_data,
                    price_history=[
                        {'date': '2024-11-01', 'price': listing_data['rent'] + 100},
                        {'date': '2024-12-01', 'price': listing_data['rent'] + 50},
                        {'date': '2025-01-01', 'price': listing_data['rent']}
                    ],
                    listing_url=f"https://apartments.com/listing/{listing_data['id']}",
                    lat=30.2672 + np.random.uniform(-0.1, 0.1),
                    lng=-97.7431 + np.random.uniform(-0.1, 0.1),
                    scraped_at=datetime.now()
                )
                listings.append(listing)
                
            return listings
            
        except Exception as e:
            logger.error(f"Error scraping Apartments.com: {e}")
            return []
    
    def scrape_zillow(self, city: str, state: str) -> List[PropertyListing]:
        """Scrape Zillow for rental listings"""
        return []
    
    def scrape_craigslist(self, city: str, state: str) -> List[PropertyListing]:
        """Scrape Craigslist for rental listings"""
        hidden_listings = [
            {
                'id': 'cl_001',
                'address': '987 Mueller District, Austin, TX',
                'rent': 2200,
                'bedrooms': 1,
                'bathrooms': 1.0,
                'sqft': 850,
                'amenities': ['wifi', 'pet', 'parking'],
                'days_on_market': 89,
                'source': 'craigslist',
                'property_manager': 'Private Owner',
                'owner_type': 'individual'
            }
        ]
        
        listings = []
        for listing_data in hidden_listings:
            listing = PropertyListing(
                **listing_data,
                price_history=[
                    {'date': '2024-10-01', 'price': listing_data['rent'] + 200},
                    {'date': '2024-11-01', 'price': listing_data['rent'] + 100},
                    {'date': '2025-01-01', 'price': listing_data['rent']}
                ],
                listing_url=f"https://craigslist.org/{listing_data['id']}",
                lat=30.2672 + np.random.uniform(-0.1, 0.1),
                lng=-97.7431 + np.random.uniform(-0.1, 0.1),
                scraped_at=datetime.now()
            )
            listings.append(listing)
            
        return listings

class MarketAnalyzer:
    """Analyze market conditions and trends"""
    
    def __init__(self):
        self.listings_data = []
        
    def store_listings(self, listings: List[PropertyListing]):
        self.listings_data = listings
        
    def analyze_market_conditions(self, city: str) -> MarketAnalysis:
        return MarketAnalysis(
            avg_days_on_market=34.0,
            price_reduction_rate=0.18,
            demand_index=72,
            seasonal_factor=0.85,
            competitive_units=23,
            median_rent=2300
        )
    
    def identify_stale_inventory(self, listings: List[PropertyListing], 
                                threshold_days: int = 45) -> List[PropertyListing]:
        return [
            listing for listing in listings 
            if listing.days_on_market >= threshold_days
        ]

class AIOfferGenerator:
    """AI-powered offer generation engine"""
    
    def __init__(self):
        self.model = None
        self.train_model()
        
    def train_model(self):
        np.random.seed(42)
        n_samples = 1000
        
        X = np.column_stack([
            np.random.randint(1, 150, n_samples),
            np.random.choice([0, 1], n_samples),
            np.random.randint(1200, 3500, n_samples),
            np.random.randint(1, 5, n_samples),
            np.random.uniform(0.7, 1.2, n_samples)
        ])
        
        y = (
            X[:, 0] * 0.002 +
            X[:, 1] * 0.05 +
            X[:, 3] * 0.03 +
            np.random.normal(0, 0.02, n_samples)
        )
        y = np.clip(y, 0, 0.25)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
    def generate_smart_offer(self, listing: PropertyListing, 
                           market_analysis: MarketAnalysis,
                           user_profile: UserProfile) -> SmartOffer:
        is_individual_owner = 1 if listing.owner_type == 'individual' else 0
        price_reductions = len(listing.price_history) - 1
        
        features = np.array([[
            listing.days_on_market,
            is_individual_owner,
            listing.rent,
            price_reductions,
            market_analysis.seasonal_factor
        ]])
        
        reduction_pct = self.model.predict(features)[0]
        recommended_price = int(listing.rent * (1 - reduction_pct))
        
        if listing.days_on_market > 60:
            strategy = "Aggressive but Fair"
        elif listing.days_on_market > 30:
            strategy = "Market-Based Negotiation"
        else:
            strategy = "Conservative Approach"
            
        base_probability = 0.6
        if listing.owner_type == 'individual':
            base_probability += 0.15
        if listing.days_on_market > 45:
            base_probability += 0.1
        if price_reductions > 0:
            base_probability += 0.08
            
        success_probability = min(0.95, base_probability)
        
        leverage_points = []
        if listing.days_on_market > 45:
            leverage_points.append(f"{listing.days_on_market} days vacant")
        if market_analysis.seasonal_factor < 1.0:
            leverage_points.append("Below market rent in winter season")
        leverage_points.append("Immediate qualified tenant")
        
        email_template = self._generate_email_template(
            listing, recommended_price, listing.rent - recommended_price, leverage_points
        )
        
        reasoning = f"""
        Property has been vacant for {listing.days_on_market} days. 
        Owner is a {listing.owner_type} likely motivated by cash flow. 
        Market analysis shows {market_analysis.price_reduction_rate:.0%} of properties 
        have reduced prices. AI model suggests {reduction_pct:.1%} reduction is optimal.
        """.strip()
        
        return SmartOffer(
            property_id=listing.id,
            recommended_price=recommended_price,
            original_price=listing.rent,
            strategy=strategy,
            success_probability=success_probability,
            leverage_points=leverage_points,
            email_template=email_template,
            reasoning=reasoning
        )
    
    def _generate_email_template(self, listing: PropertyListing, 
                                offer_price: int, savings: int,
                                leverage_points: List[str]) -> str:
        return f"""Subject: Immediate Lease Opportunity - {listing.address}

Dear Property Manager,

I hope this email finds you well. I'm writing regarding the {listing.bedrooms}BR/{listing.bathrooms}BA unit at {listing.address}.

**Offer Details:**
• Monthly Rent: ${offer_price:,}
• Lease Term: 12 months 
• Move-in Date: Within 30 days

**Why This Works for You:**
{chr(10).join(f'• {point}' for point in leverage_points)}

Best regards,
[Your Name]"""

class ApartmentIQ:
    """Main ApartmentIQ platform class"""
    
    def __init__(self):
        self.scraper = ApartmentScraper()
        self.analyzer = MarketAnalyzer()
        self.offer_generator = AIOfferGenerator()
        
    def discover_opportunities(self, city: str, state: str) -> List[PropertyListing]:
        all_listings = []
        all_listings.extend(self.scraper.scrape_apartments_com(city, state))
        all_listings.extend(self.scraper.scrape_zillow(city, state))
        all_listings.extend(self.scraper.scrape_craigslist(city, state))
        self.analyzer.store_listings(all_listings)
        return all_listings
    
    def analyze_and_generate_offers(self, listings: List[PropertyListing],
                                   user_profile: UserProfile) -> List[SmartOffer]:
        market_analysis = self.analyzer.analyze_market_conditions("Austin")
        stale_listings = self.analyzer.identify_stale_inventory(listings)
        
        offers = []
        for listing in stale_listings:
            if (listing.rent <= user_profile.max_budget and 
                listing.bedrooms >= user_profile.min_bedrooms):
                offers.append(self.offer_generator.generate_smart_offer(
                    listing, market_analysis, user_profile
                ))
        
        offers.sort(key=lambda x: (x.success_probability, 
                                 x.original_price - x.recommended_price), 
                   reverse=True)
        return offers
    
    def run_analysis(self, city: str = "Austin", state: str = "TX") -> Dict[str, Any]:
        user_profile = UserProfile(
            current_rent=2600,
            lease_expires="March 2025",
            max_budget=2500,
            work_lat=30.2672,
            work_lng=-97.7431,
            preferred_amenities=['gym', 'pet', 'parking'],
            min_bedrooms=1,
            max_commute_time=30
        )
        
        listings = self.discover_opportunities(city, state)
        offers = self.analyze_and_generate_offers(listings, user_profile)
        
        total_potential_savings = sum(
            offer.original_price - offer.recommended_price 
            for offer in offers
        )
        
        avg_success_rate = np.mean([offer.success_probability for offer in offers]) if offers else 0
        
        return {
            'total_listings': len(listings),
            'hidden_opportunities': len([l for l in listings if l.source == 'craigslist']),
            'smart_offers_generated': len(offers),
            'total_potential_monthly_savings': total_potential_savings,
            'avg_success_probability': avg_success_rate,
            'top_offers': [asdict(offer) for offer in offers[:3]]
        }

def streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(page_title="ApartmentIQ", layout="wide")
    
    st.title("🏠 ApartmentIQ: AI-Powered Rental Negotiation")
    st.markdown("Discover hidden rental opportunities and generate AI-optimized offers")
    
    with st.sidebar:
        st.header("Your Preferences")
        city = st.text_input("City", "Austin")
        state = st.text_input("State", "TX")
        max_budget = st.number_input("Max Budget ($)", 1000, 10000, 2500)
        min_bedrooms = st.number_input("Min Bedrooms", 1, 5, 1)
        analyze_btn = st.button("Analyze Market", type="primary")
    
    if analyze_btn:
        with st.spinner("🚀 Scanning rental market for hidden opportunities..."):
            apartment_iq = ApartmentIQ()
            user_profile = UserProfile(
                current_rent=max_budget + 100,
                lease_expires="March 2025",
                max_budget=max_budget,
                work_lat=30.2672,
                work_lng=-97.7431,
                preferred_amenities=['gym', 'pet', 'parking'],
                min_bedrooms=min_bedrooms,
                max_commute_time=30
            )
            results = apartment_iq.run_analysis(city, state)
        
        st.success("Analysis complete!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Listings", results['total_listings'])
        col2.metric("Hidden Opportunities", results['hidden_opportunities'])
        col3.metric("Potential Monthly Savings", 
                   f"${results['total_potential_monthly_savings']:,}")
        
        st.subheader("💡 Top AI-Generated Offers")
        for i, offer in enumerate(results['top_offers'], 1):
            savings = offer['original_price'] - offer['recommended_price']
            
            with st.expander(f"Offer #{i}: ${offer['recommended_price']:,} (Save ${savings:,})"):
                cols = st.columns([1,2])
                cols[0].metric("Original Price", f"${offer['original_price']:,}")
                cols[0].metric("Your Savings", f"${savings:,}", 
                              delta=f"{savings/offer['original_price']:.1%}")
                cols[1].progress(offer['success_probability'], 
                               text=f"Success Probability: {offer['success_probability']:.1%}")
                
                st.markdown("**Strategy:** " + offer['strategy'])
                st.markdown("**Key Leverage Points:**")
                for point in offer['leverage_points']:
                    st.markdown(f"- {point}")
                
                with st.expander("📧 View Email Template"):
                    st.code(offer['email_template'])

if __name__ == "__main__":
    streamlit_app()
