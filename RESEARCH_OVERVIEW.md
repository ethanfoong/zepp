# Extreme Coastal Climatology: A Research Journey
## From Ocean Buoys to Understanding Temperature Extremes

**Project Lead**: UC Berkeley Prof. Zeppetello  
**Focus**: Understanding extreme coastal temperature variations  
**Current Phase**: Visualizing comparative analysis of ocean vs. land observations

---

## The Big Picture

Our research asks a deceptively simple question: **How do extreme temperatures behave at the ocean-land interface?**

To answer this, we've built an entire workflow that:
1. Gathers decades of temperature observations from buoys scattered across the world's oceans
2. Compares those with measurements from nearby land-based weather stations
3. Analyzes statistical patterns in how these temperatures vary
4. Creates maps and charts showing what we've discovered

This document walks through how we got here—the journey from raw data to meaningful insights.

---

## Part 1: Getting the Data

### Finding the Buoys

The National Oceanic and Atmospheric Administration (NOAA) maintains a network of buoys that collect meteorological data across all major oceans. These aren't just temperature readings—they record air temperature, water temperature, wind, pressure, and more at regular intervals.

**What we did:**
- Discovered that NOAA has an online archive of historical buoy data going back decades
- Built a web scraper to identify which buoys have sufficient historical records (we focused on those with at least 10 years of continuous data)
- Located approximately 120 buoys around the world's coasts that met our criteria
- Extracted their geographic coordinates so we could place them on maps later

**Why it matters:**
Without this structured discovery process, we'd be manually hunting for data. The automated approach lets us work with a standardized dataset across all oceans.

### Gathering Land Observations

The contrast we're investigating requires parallel land data. We pulled from NOAA's land-based climate records, which include:
- Daily maximum temperatures (TMAX)
- Daily minimum temperatures (TMIN)
- Computed statistics like variance and skewness

The idea: **Can we match each ocean buoy with nearby land stations to make fair comparisons?**

---

## Part 2: Processing the Raw Data

### The Download & Cache Strategy

Historical buoy data comes compressed in thousands of individual files—one per year per buoy. Downloading everything fresh every time we run an analysis would be inefficient.

**What we implemented:**
- A smart caching system that downloads files once and stores them locally
- Automatic decompression and parsing of NOAA's standardized data format
- Checks to ensure data quality (handling missing values, inconsistent formats)
- Multi-threaded concurrent downloads for speed

**Translation:** Instead of waiting hours for data downloads repeatedly, we now wait once and can reuse the data instantly.

### Cleaning & Standardizing

Raw buoy data isn't immediately ready for analysis. It contains:
- Multiple temperature measurements per day (NOAA records observations hourly or every few hours)
- Occasional gaps or missing values (marked with placeholder values like 999.0)
- A quirk with leap years that we needed to handle consistently

**Our approach:**
- Convert all compressed files into clean, standardized dataframes
- Aggregate hourly observations into daily values
- Remove February 29 on leap years to maintain consistent 365-day years
- Ensure all files have identical column structures

**Result:** Standardized, clean datasets ready for statistical analysis.

---

## Part 3: The Core Analysis — Warm Season Anomalies

### What is a "Warm Season"?

This is where our research gets interesting. We don't analyze temperature extremes for the entire year because seasonal patterns dominate. Instead, we focus on each location's **warm season**—roughly the period of 2-3 months when temperatures are at their peak.

**How we identify it:**
1. For each buoy, we compute the average temperature for each day-of-year (using all available years)
2. We find the day when this average peaks (the climatological maximum)
3. We look at a 100-day window centered on this peak—that's our warm season

**Why 100 days?** It's a balance—wide enough to capture meaningful warm season variability, narrow enough to focus on when extremes actually matter.

### Computing Anomalies

Here's the key insight: we don't study raw temperatures. Instead, we study **deviations from the expected pattern**.

**The calculation:**
- For each day in the warm season, we subtract the long-term average for that day
- What remains is the "anomaly"—how much warmer or cooler than usual
- We do this for every day across all 10+ years of data

**Why this matters:** This removes the obvious temperature pattern (it's hotter in July than January) and lets us focus on the interesting question: *when is it unusually warm or cold for the season?*

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

**Why standardize?** It ensures our work is reproducible and can be used by collaborators without worrying about proprietary formats.

---

## Part 5: Making Maps

### From Data to Geography

Raw numbers don't tell a story the way maps do. So we created visualizations:

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

These charts answer: *How different is the ocean from nearby land?*

---

## Part 6: Putting It All Together

### The Automated Pipeline

We didn't want to manually run dozens of scripts every time we needed fresh results. So we built an automated workflow:

1. **Discovery**: Scan NOAA's servers to find all available buoys
2. **Filtering**: Keep only those with 10+ years of data
3. **Processing**: Download, clean, and process each buoy (in parallel for speed)
4. **Analysis**: Compute variance and skewness
5. **Export**: Save results to NetCDF files and aggregated cache
6. **Visualization**: Generate all maps and charts
7. **Quality Control**: Tests to verify everything worked correctly

**Runtime:** A few hours for ~120 buoys on modern hardware. Once cached, re-runs with updates are much faster.

### The Notebook Investigations

Alongside automated pipelines, we have exploratory notebooks where we:
- Test new ideas and methods
- Deep-dive into specific buoys
- Create detailed case studies
- Experiment with different window sizes, metrics, and definitions

These notebooks are where discovery happens. The pipelines are where repeatability and scale are achieved.

---

## Part 7: From PNG to PDF

Our visualizations are currently saved as high-resolution images. The next logical step:

**Compile into a research report PDF** containing:
- Executive summary of findings
- Methodology explanation
- All global and regional maps
- Statistical comparison charts
- Data quality metrics
- Conclusions and interpretations

This creates a publication-ready document that can be shared with collaborators, submitted to conferences, or included in papers.

---

## The Workflow at a Glance

```
Raw NOAA Buoy Data          Land Weather Observations
          ↓                              ↓
    Web Scraping & Download      Land Climate Records
          ↓                              ↓
      Clean & Cache               Align & Process
          ↓                              ↓
    Compute Warm Seasons          Compute Statistics
          ↓                              ↓
    Calculate Anomalies ←────→ Compare Ocean vs. Land
          ↓                              ↓
    Variance & Skewness      Statistical Comparison
          ↓                              ↓
    Export to NetCDF          Create Visualizations
          ↓                              ↓
    Create Maps & Charts       Generate Figures
          ↓                              ↓
         PDF Report (Future)
```

---

## Key Insights So Far

**What we've discovered:**

1. **Not all oceans are alike:** Some regions show high temperature variability; others are remarkably stable.

2. **Ocean vs. Land differences:** Ocean temperatures tend to be more moderate in extremes compared to nearby land—likely due to water's high heat capacity.

3. **Geographic patterns:** Certain coasts (e.g., western coasts with cold currents) show different extreme patterns than others.

4. **Skewness variations:** Some regions show a bias toward heat extremes; others toward cold extremes. This has implications for heatwave and frost risk.

These aren't final conclusions—they're observations that guide where to dig deeper and what questions matter most.

---

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

---

## Why This Approach?

### Scalability
We started with one or two buoys. Now we handle 120+ automatically. The infrastructure scales without rewriting code.

### Reproducibility
Every step is documented—from the raw NOAA source to the final figure. Anyone can re-run the analysis and get identical results.

### Efficiency
Caching, parallel processing, and automation mean we can iterate quickly without waiting hours for downloads and processing.

### Quality
Automated tests and validation checks catch errors early. The same analysis is applied consistently to all buoys.

### Extensibility
New analyses can build on existing infrastructure. Want to test a different window size? Change one parameter. Want to add more buoys? The pipeline handles it automatically.

---

## What's Next?

**Immediate priorities:**
1. Compile all visualizations into publication-ready PDFs
2. Identify and fix remaining code redundancies
3. Expand land-station network for deeper comparisons

**Longer-term goals:**
1. **Trend detection:** Are extremes changing over time?
2. **Attribution:** Can we connect extremes to large-scale climate patterns?
3. **Prediction:** Can machine learning forecast future extremes?
4. **Validation:** Compare our observations to climate models

---

## In Summary

We've built a complete research pipeline that answers real scientific questions about how extreme temperatures behave at ocean-land interfaces. The journey from NOAA's raw data servers to publication-quality maps and statistical comparisons involves data engineering, statistical analysis, visualization, and software automation working in concert.

The workflow is designed to be:
- **Automated** so we can process hundreds of data sources consistently
- **Reproducible** so others can verify and build on our work
- **Extensible** so we can add new analyses and comparisons
- **Rigorous** so we can trust the results

This foundation positions us well for deeper investigations into coastal climate extremes and their implications for understanding global temperature dynamics.

---

**Document Updated**: April 29, 2026  
**Status**: Active Research — Ready for PDF compilation phase
