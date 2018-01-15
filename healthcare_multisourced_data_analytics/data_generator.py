# -*- coding: utf-8 -*-
"""
Pakistani Advertisement Data Generator
Compatible with Python 3.6+ (2018 standards)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from config import Config

class PakistaniAdDataGenerator:
    """Generate realistic Pakistani advertisement data"""
    
    def __init__(self, num_records: int = 1000):
        self.num_records = num_records
        self.config = Config()
        random.seed(42)  # For reproducible results
        np.random.seed(42)
    
    def generate_pakistani_names(self) -> List[str]:
        """Generate Pakistani names for ad topic lines"""
        pakistani_products = [
            'Shan Masala', 'National Foods', 'Tapal Tea', 'Lipton Pakistan',
            'Olpers Milk', 'Nestle Pakistan', 'Unilever Pakistan', 'P&G Pakistan',
            'Mobilink Jazz', 'Telenor Pakistan', 'Zong 4G', 'Ufone',
            'Bank Alfalah', 'HBL Pakistan', 'MCB Bank', 'UBL Digital',
            'Daraz Pakistan', 'Foodpanda Pakistan', 'Careem Pakistan', 'Uber Pakistan',
            'Gul Ahmed', 'Khaadi Pakistan', 'Sapphire Pakistan', 'Alkaram Studio'
        ]
        
        ad_templates = [
            'Premium {} - Limited Time Offer',
            'New {} Launch - Special Discount',
            'Best {} Deals in Pakistan',
            'Exclusive {} Collection 2018',
            'Top Quality {} - Free Delivery',
            'Latest {} Technology - Buy Now',
            'Authentic {} - Nationwide Service',
            'Professional {} Solutions Pakistan'
        ]
        
        ad_topics = []
        for _ in range(self.num_records):
            product = random.choice(pakistani_products)
            template = random.choice(ad_templates)
            ad_topics.append(template.format(product))
        
        return ad_topics
    
    def generate_income_by_city(self, city: str) -> float:
        """Generate realistic income based on Pakistani city"""
        if city in self.config.CITY_INCOME_RANGES:
            min_income, max_income = self.config.CITY_INCOME_RANGES[city]
        else:
            min_income, max_income = (25000, 75000)  # Default range
        
        # Generate income with normal distribution
        mean_income = (min_income + max_income) / 2
        std_income = (max_income - min_income) / 6
        income = np.random.normal(mean_income, std_income)
        
        return max(min_income, min(max_income, income))
    
    def generate_timestamp(self) -> str:
        """Generate realistic timestamp for 2018"""
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2018, 12, 31)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        random_date = start_date + timedelta(days=random_days)
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        
        final_datetime = random_date.replace(
            hour=random_hour, 
            minute=random_minute, 
            second=random_second
        )
        
        return final_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    def calculate_click_probability(self, age: int, income: float, 
                                  internet_usage: float, time_on_site: float) -> float:
        """Calculate click probability based on user characteristics"""
        # Base probability
        prob = 0.3
        
        # Age factor (25-35 age group more likely to click)
        if 25 <= age <= 35:
            prob += 0.2
        elif age < 25 or age > 45:
            prob -= 0.1
        
        # Income factor (middle income more likely to click)
        if 40000 <= income <= 80000:
            prob += 0.15
        elif income > 100000:
            prob -= 0.05
        
        # Internet usage factor
        if internet_usage > 200:
            prob += 0.1
        elif internet_usage < 100:
            prob -= 0.1
        
        # Time on site factor
        if time_on_site > 60:
            prob += 0.15
        elif time_on_site < 30:
            prob -= 0.1
        
        return max(0, min(1, prob))
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete Pakistani advertisement dataset"""
        print("Generating Pakistani advertisement dataset...")
        
        data = {
            'Daily Time Spent on Site': [],
            'Age': [],
            'Area Income': [],
            'Daily Internet Usage': [],
            'Ad Topic Line': [],
            'City': [],
            'Male': [],
            'Country': [],
            'Timestamp': [],
            'Clicked on Ad': []
        }
        
        ad_topics = self.generate_pakistani_names()
        
        for i in range(self.num_records):
            # Generate basic demographics
            age = np.random.randint(18, 65)
            city = random.choice(self.config.PAKISTANI_CITIES)
            income = self.generate_income_by_city(city)
            gender = random.choice([0, 1])  # 0: Female, 1: Male
            
            # Generate internet behavior
            daily_internet_usage = np.random.normal(180, 50)
            daily_internet_usage = max(50, min(300, daily_internet_usage))
            
            time_on_site = np.random.normal(65, 20)
            time_on_site = max(10, min(120, time_on_site))
            
            # Calculate click probability and determine click
            click_prob = self.calculate_click_probability(
                age, income, daily_internet_usage, time_on_site
            )
            clicked = 1 if random.random() < click_prob else 0
            
            # Add to dataset
            data['Daily Time Spent on Site'].append(round(time_on_site, 2))
            data['Age'].append(age)
            data['Area Income'].append(round(income, 2))
            data['Daily Internet Usage'].append(round(daily_internet_usage, 2))
            data['Ad Topic Line'].append(ad_topics[i])
            data['City'].append(city)
            data['Male'].append(gender)
            data['Country'].append('Pakistan')
            data['Timestamp'].append(self.generate_timestamp())
            data['Clicked on Ad'].append(clicked)
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records for Pakistani market")
        return df
    
    def save_dataset(self, filename: str = None) -> str:
        """Generate and save the dataset"""
        if filename is None:
            filename = 'data/advertising_pakistan.csv'
        
        df = self.generate_dataset()
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Click rate: {df['Clicked on Ad'].mean():.2%}")
        print(f"Average age: {df['Age'].mean():.1f}")
        print(f"Average income: PKR {df['Area Income'].mean():,.0f}")
        print(f"Cities covered: {df['City'].nunique()}")
        
        return filename

if __name__ == "__main__":
    generator = PakistaniAdDataGenerator(1000)
    generator.save_dataset()