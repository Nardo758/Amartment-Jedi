#!/usr/bin/env python3
"""
ApartmentIQ: AI-Powered Vacancy & Relocation Platform
Core data processing and analysis engine - Streamlit Cloud Compatible
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import re
from bs4 import BeautifulSoup
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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
    price_history: List[Dict]
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
            # Mock data for demonstration - replace with real scraping in production
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
        logger.info(f"Scraping Zillow for {city}, {state}")
        return []
    
    def scrape_craigslist(self, city: str, state: str) -> List[PropertyListing]:
        """Scrape Craigslist for rental listings"""
        logger.info(f"Scraping Craigslist for {city}, {state}")
        
        # Mock hidden inventory from Craigslist
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
        # Use in-memory database for Streamlit Cloud compatibility
        self.db_path = ':memory:'
        self.init_database()
        
    def init_database(self):
        """Initialize in-memory SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS listings (
                id TEXT PRIMARY KEY,
                address TEXT,
                rent INTEGER,
                bedrooms INTEGER,
                bathrooms REAL,
                sqft INTEGER,
                amenities TEXT,
                days_on_market INTEGER,
                source TEXT,
                owner_type TEXT,
                scraped_at TIMESTAMP,
                lat REAL,
                lng REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                listing_id TEXT,
                date TEXT,
                price INTEGER,
                FOREIGN KEY(listing_id) REFERENCES listings(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_listings(self, listings: List[PropertyListing]):
        """Store scraped listings in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for listing in listings:
            cursor.execute('''
                INSERT OR REPLACE INTO listings VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                listing.id, listing.address, listing.rent, listing.bedrooms,
                listing.bathrooms, listing.sqft, json.dumps(listing.amenities),
                listing.days_on_market, listing.source, listing.owner_type,
                listing.scraped_at, listing.lat, listing.lng
            ))
            
            for price_point in listing.price_history:
                cursor.execute('''
                    INSERT OR REPLACE INTO price_history VALUES (?, ?, ?)
                ''', (listing.id, price_point['date'], price_point['price']))
                
        conn.commit()
        conn.close()
        
    def analyze_market_conditions(self, city: str) -> MarketAnalysis:
        """Analyze current market conditions"""
        # Return realistic market analysis data
        return MarketAnalysis(
            avg_days_on_market=34.0,
            price_reduction_rate=0.18,
            demand_index=72,
            seasonal_factor=0.85,  # Winter reduction
            competitive_units=23,
            median_rent=2300
        )
    
    def identify_stale_inventory(self, listings: List[PropertyListing], 
                                threshold_days: int = 45) -> List[PropertyListing]:
        """Identify properties that have been on market too long"""
        stale_listings = [
            listing for listing in listings 
            if listing.days_on_market >= threshold_days
        ]
        
        logger.info(f"Found {len(stale_listings)} stale inventory opportunities")
        return stale_listings

class AIOfferGenerator:
    """AI-powered offer generation engine"""
    
    def __init__(self):
        self.model = None
        self.train_model()
        
    def train_model(self):
        """Train the pricing model on historical data"""
        # Mock training data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        X = np.column_stack([
            np.random.randint(1, 150, n_samples),  # days_on_market
            np.random.choice([0, 1], n_samples),   # is_individual_owner
            np.random.randint(1200, 3500, n_samples),  # market_rent
            np.random.randint(1, 5, n_samples),    # price_reductions
            np.random.uniform(0.7, 1.2, n_samples)  # seasonal_factor
        ])
        
        # Target: percentage reduction from asking price
        y = (
            X[:, 0] * 0.002 +  # days on market effect
            X[:, 1] * 0.05 +   # individual owner more flexible
            X[:, 3] * 0.03 +   # previous reductions indicate flexibility
            np.random.normal(0, 0.02, n_samples)  # noise
        )
        y = np.clip(y, 0, 0.25)  # Cap at 25% reduction
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        logger.info("AI pricing model trained successfully")
    
    def generate_smart_offer(self, listing: PropertyListing, 
                           market_analysis: MarketAnalysis,
                           user_profile: UserProfile) -> SmartOffer:
        """Generate AI-powered offer for a property"""
        
        # Prepare features for model
        is_individual_owner = 1 if listing.owner_type == 'individual' else 0
        price_reductions = len(listing.price_history) - 1
        
        features = np.array([[
            listing.days_on_market,
            is_individual_owner,
            listing.rent,
            price_reductions,
            market_analysis.seasonal_factor
        ]])
        
        # Predict optimal reduction percentage
        reduction_pct = self.model.predict(features)[0]
        recommended_price = int(listing.rent * (1 - reduction_pct))
        
        # Calculate carrying cost and owner motivation
        daily_carrying_cost = listing.rent * 0.03 / 30  # 3% of rent per month
        total_loss = daily_carrying_cost * listing.days_on_market
        
        # Determine strategy based on market conditions
        if listing.days_on_market > 60:
            strategy = "Aggressive but Fair"
        elif listing.days_on_market > 30:
            strategy = "Market-Based Negotiation"
        else:
            strategy = "Conservative Approach"
            
        # Calculate success probability
        base_probability = 0.6
        if listing.owner_type == 'individual':
            base_probability += 0.15
        if listing.days_on_market > 45:
            base_probability += 0.1
        if price_reductions > 0:
            base_probability += 0.08
            
        success_probability = min(0.95, base_probability)
        
        # Generate leverage points
        leverage_points = []
        if listing.days_on_market > 45:
            leverage_points.append(f"{listing.days_on_market} days vacant = ${int(total_loss):,} in lost revenue")
        if market_analysis.seasonal_factor < 1.0:
            leverage_points.append("Below market rent in winter season")
        leverage_points.append("Immediate qualified tenant")
        leverage_points.append("No agent commission needed")
        
        # Generate email template
        email_template = self._generate_email_template(
            listing, recommended_price, listing.rent - recommended_price, leverage_points
        )
        
        # Generate reasoning
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
        """Generate personalized email template"""
        
        template = f"""Subject: Immediate Lease Opportunity - {listing.address}

Dear Property Manager,

I hope this email finds you well. I'm writing regarding the {listing.bedrooms}BR/{listing.bathrooms}BA unit at {listing.address} that has been available.

I'm a qualified tenant looking to secure housing immediately and would like to present a competitive offer:

**Offer Details:**
• Monthly Rent: ${offer_price:,}
• Lease Term: 12 months 
• Move-in Date: Within 30 days
• Security Deposit: 1 month rent
• No pets, non-smoker

**Why This Works for You:**
{chr(10).join(f'• {point}' for point in leverage_points)}

I understand the market has been challenging, and I believe this offer reflects a fair value that benefits us both. I'm happy to provide income verification, references, and can sign a lease this week.

Would you be available for a brief call to discuss this opportunity?

Best regards,
[Your Name]
[Phone] | [Email]

P.S. I'm also happy to consider a longer lease term if that would be beneficial."""

        return template

class ApartmentIQ:
    """Main ApartmentIQ platform class - Streamlit Cloud Compatible"""
    
    def __init__(self):
        self.scraper = ApartmentScraper()
        self.analyzer = MarketAnalyzer()
        self.offer_generator = AIOfferGenerator()
        
    def discover_opportunities(self, city: str, state: str) -> List[PropertyListing]:
        """Discover apartment opportunities across multiple sources"""
        logger.info(f"Discovering opportunities in {city}, {state}")
        
        all_listings = []
        
        # Scrape multiple sources
        apartments_com_listings = self.scraper.scrape_apartments_com(city, state)
        all_listings.extend(apartments_com_listings)
        
        zillow_listings = self.scraper.scrape_zillow(city, state)
        all_listings.extend(zillow_listings)
        
        craigslist_listings = self.scraper.scrape_craigslist(city, state)
        all_listings.extend(craigslist_listings)
        
        # Store in database
        self.analyzer.store_listings(all_listings)
        
        logger.info(f"Found {len(all_listings)} total listings")
        return all_listings
    
    def analyze_and_generate_offers(self, listings: List[PropertyListing],
                                   user_profile: UserProfile) -> List[SmartOffer]:
        """Analyze market and generate smart offers"""
        
        # Analyze market conditions
        market_analysis = self.analyzer.analyze_market_conditions("Austin")
        logger.info(f"Market analysis complete: {market_analysis.demand_index}/100 demand index")
        
        # Identify stale inventory (hidden opportunities)
        stale_listings = self.analyzer.identify_stale_inventory(listings)
        
        # Generate smart offers for promising properties
        offers = []
        for listing in stale_listings:
            # Filter by user preferences
            if (listing.rent <= user_profile.max_budget and 
                listing.bedrooms >= user_profile.min_bedrooms):
                
                offer = self.offer_generator.generate_smart_offer(
                    listing, market_analysis, user_profile
                )
                offers.append(offer)
        
        # Sort by success probability and savings potential
        offers.sort(key=lambda x: (x.success_probability, 
                                  x.original_price - x.recommended_price), 
                   reverse=True)
        
        logger.info(f"Generated {len(offers)} smart offers")
        return offers
    
    def run_analysis(self, city: str = "Austin", state: str = "TX") -> Dict:
        """Run complete ApartmentIQ analysis"""
        
        # Sample user profile
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
        
        # Discover opportunities
        listings = self.discover_opportunities(city, state)
        
        # Generate offers
        offers = self.analyze_and_generate_offers(listings, user_profile)
        
        # Calculate summary statistics
        total_potential_savings = sum(
            offer.original_price - offer.recommended_price 
            for offer in offers
        )
        
        avg_success_rate = np.mean([offer.success_probability for offer in offers]) if offers else 0
        
        results = {
            'total_listings': len(listings),
            'hidden_opportunities': len([l for l in listings if l.source == 'craigslist']),
            'smart_offers_generated': len(offers),
            'total_potential_monthly_savings': total_potential_savings,
            'avg_success_probability': avg_success_rate,
            'top_offers': [asdict(offer) for offer in offers[:3]]
        }
        
        return results

def main():
    """Main execution function"""
    logger.info("Starting ApartmentIQ analysis...")
    
    # Initialize platform
    apartment_iq = ApartmentIQ()
    
    # Run analysis
    results = apartment_iq.run_analysis()
    
    # Print results
    print("\n" + "="*60)
    print("APARTMENTIQ ANALYSIS RESULTS")
    print("="*60)
    print(f"📊 Total listings analyzed: {results['total_listings']}")
    print(f"🔍 Hidden opportunities found: {results['hidden_opportunities']}")
    print(f"🎯 Smart offers generated: {results['smart_offers_generated']}")
    print(f"💰 Total potential monthly savings: ${results['total_potential_monthly_savings']:,}")
    print(f"📈 Average success probability: {results['avg_success_probability']:.1%}")
    
    print("\n🏆 TOP OPPORTUNITIES:")
    print("-" * 40)
    
    for i, offer in enumerate(results['top_offers'], 1):
        savings = offer['original_price'] - offer['recommended_price']
        print(f"{i}. Property #{offer['property_id']}")
        print(f"   💵 Asking: ${offer['original_price']:,} → Offer: ${offer['recommended_price']:,}")
        print(f"   💡 Monthly savings: ${savings:,}")
        print(f"   📊 Success probability: {offer['success_probability']:.1%}")
        print(f"   🎯 Strategy: {offer['strategy']}")
        print()

if __name__ == "__main__":
    main()
