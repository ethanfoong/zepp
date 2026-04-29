# Extreme Coastal Climatology Research 
## Understanding Temperature Extremes at an Ocean Buoy and Land Tower Scale

**Project Lead**: UC Berkeley Prof. Lucas Vargas Zeppetello
**Reseacher**: Ethan Foong
**Focus**: Understanding extreme coastal temperature anomalies & their respective variations  
**Current Phase**: Visualizing comparative analysis of ocean vs. land observations

This repository contains tools, analyses, and data workflows developed as part of Prof. Zeppetello’s research on extreme coastal temperature variations. The project focuses on understanding and modeling long-term global climatological extremes through data-driven statistical analysis of land tower temperature measurements and oceanic buoy atmospheric temperatures. 

---

## At a High Level 

Our research asks a deceptively overlooked curiousity: **How are extreme temperatures categorized at the coast with high temperature disparities between land and the ocean?**

To answer this, we've built an entire workflow that:
1. Gathers decades of temperature observations from coastal buoys scattered across the United States
2. Compares those with observations from nearby land tower stations
3. Analyzes statistical patterns in how these temperatures vary
4. Creates maps and charts showing what we've discovered

This README walks through how we were able to draw meaningful implications on extreme temperatures found in coastal climatology from unfiltered datasets. 

---

## Part 1: Aggregating the Data

### Finding the Buoys

The National Oceanic and Atmospheric Administration (NOAA) maintains a network of buoys that collect meteorological data across all major oceans. These buoys span decades and contain data at high granularity ranging from air temperature, water temperature, wind, pressure, and more at regular intervals.

**Our Approach:**
- Discovered that NOAA has an online archive of historical buoy data going back decades
- Built a web scraper to identify which buoys have sufficient historical records (we focused on those with at least 10 years of continuous data)
- Located approximately ~120 buoys around the world's coasts that met our criteria
- Extracted the geographic coordinates from the buoy location dataset to pinpoint each buoy's data to their respective location on the coast through Cartopy

### Gathering Land Observations

This part was gathered by Prof. Zeppetello himself (will talk to him more about it on Wednesday).

The contrast we're investigating requires parallel land data. We pulled this data from land-based climate records (specify which records) , which include:
- Daily maximum temperatures (TMAX)
- Daily minimum temperatures (TMIN)
- Daily temperature variance (TVAR)
- Daily temperature skewness (TSKEW)

To sum up, the idea we're exploring is: **How much do extreme temperature anomalies differ in ocean buoys and their nearby land tower observations?**

---

## Part 2: Processing the Raw Data

### The Download & Cache Strategy

Historical buoy data comes compressed with thousands of line to register each hour of each year per buoy. Freshly sraping all the data from the NOAA proved to be inefficient and computationally expensive, therefore I created a workflow that was able to scrape all buoy data and store its cache for later use. 

**Implementation Strategy:**
- A smart caching system that downloads files once and stores them locally
- Automatic decompression and parsing of NOAA's standardized data format
- Exploratory data analysis to ensure data quality and completeness (90+%)
- missing value and inconsistent format handling across messy data from ocean buoy files
- Multi-threaded concurrent downloads for speed

**Usage Significance** Instead of waiting hours for data downloads repeatedly, we now wait once and can reuse the data instantly (which was great for visualization!)

### Cleaning & Standardizing

As I previously noted, raw buoy data isn't immediately ready for analysis and is not standardized in its file structure. Common themes I noticed were:
- Inconsistent temperature measurements per day (NOAA records observations hourly or every few hours)
- Occasional gaps or missing values (marked with placeholder values like 999.0 or 0)
- Logging February 29th on Leap Years which throws off the standard 365 day observation cycle
  
**Conducting the Data Cleaning:**
- Convert all compressed files into clean, standardized dataframes
- Aggregate hourly observations into daily values
- Remove February 29 on leap years to maintain consistent 365-day years
- Ensure all files have identical column structures

---

## Part 3: The Core Analysis — Warm Season Anomalies

### What is a "Warm Season"?

In our research, we did not take into account temperature extremes for the entire year but instead only analyzed each year's **"warm season"** which spans each years warmest 100 day cycle - in another words: the period of 2-3 months when temperatures are at their peak.

**Segmenting Out the Warm Cycle:**
1. For each buoy, we compute the average temperature for each day-of-year (using all available years)
2. We find the day when this average peaks (the climatological maximum)
3. We look at a 100-day window centered on this peak—that's our warm season

**100 Day Significance** To capture meaningful warm season variability, we want to analyze a seasonal period narrow enough to focus on when extremes actually matter and are conducive of ___________

### Computing Anomalies

Computing anomalies requires understanding how much temperatures deviate from their climatological mean: 

**The calculation:**
- For each day in the warm season, we subtract the long-term average for that day
- What remains is the "anomaly"—how much warmer or cooler than usual
- We do this for every day across all 10+ years of data

**Why this matters:** This removes the assumption of a regular temperature pattern (it's hotter in July than January) and instead hones us in the mindset of thinking about: when is it unusually warm or cold for the season?

### Computing Statistics

Once we have anomalies, we compute two key statistics:

1. **Variance**: How spread-out are these anomalies? A buoy with high variance has wild temperature swings. A buoy with low variance has predictable temperatures.

2. **Skewness**: Is the distribution lopsided? 
   - Positive skewness = more extreme warm events (heat waves are worse than cold snaps)
   - Negative skewness = more extreme cold events
   - Near zero = symmetric extremes

These two statistics become our window into how extreme temperatures behave differently across the globe.

### Comparing Ocean to Land

For each buoy, we find nearby land stations and compute the same statistics on their data. Then we ask:

- Do ocean temperatures vary more or less than land temperatures?
- Are ocean temperature extremes more biased toward heat or cold?
- How does this pattern differ by geographic region?

This comparison reveals fundamental differences in how thermal extremes behave at the ocean-land interface.

---

## Part 4: Storing Results

We export our computed statistics into NetCDF format—a scientific standard that other researchers, tools, and programming languages can easily read.

**What gets stored:**
- The anomalies themselves (the complete time series of warm-season deviations)
- The long-term statistics we computed (variance, skewness)
- The seasonal climatology (the average temperature pattern we used as baseline)
- Metadata (coordinates, years of data, etc.)

---

## Part 5: Making Maps

### From Data to Geography

Raw numbers do not always properly or comprehensive tell a story the way visualizations do.

**Global Maps:**
- World map showing every buoy colored by its variance (variability in warm-season temperatures)
- World map showing every buoy colored by its skewness (warm vs. cold extremes)
- Regional zooms (e.g., West Coast, Atlantic, Gulf of Mexico) with higher detail

These maps immediately reveal patterns: *Are certain ocean regions more extreme than others? Do extremes cluster in specific areas?*

### Comparative Analysis Charts

We also created histograms and probability density plots comparing buoy statistics to land observations:
- Overlaid distributions showing buoy vs. land variance
- Side-by-side comparisons by coast (West Coast, Atlantic, etc.)
- Scatter plots showing correlation between ocean and land extremes


## Part 6: Putting It All Together

### The Automated Pipeline


## How We Organized the Work

**Code & Functions:**
- Helper modules for downloading, parsing, and processing buoy data
- Statistical analysis functions for computing anomalies and metrics
- Visualization functions for creating maps and charts
- Automated pipeline scripts that orchestrate everything

**Data Storage:**
- **Cache folder:** Stores downloaded files and intermediate results (don't re-download)
- **NetCDF folder:** Standardized output files (one per buoy)
- **Figures folder:** Generated maps, charts, and visualizations

**Notebooks:**
- Exploratory investigations and case studies
- Where we test ideas before automating them
- Detailed documentation of methodology

**Document Updated**: April 29, 2026  

