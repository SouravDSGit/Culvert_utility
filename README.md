# CULVERT Web-App - Utility Functions

A collection of utility functions and modules supporting the Climate and Upland Loading Vulnerability Evaluation and Risk Analysis Tool (CULVERT) web application for hydrologic risk assessment and infrastructure vulnerability analysis.

## About

This repository contains the core utility functions, calculation modules, and helper scripts that power the CULVERT web application. These utilities enable engineers and researchers to assess the hydrologic and geomorphologic vulnerability of culverts, fords, and bridges across the United States.

## What This Repository Supports

### Watershed Analysis
- Sub-meter resolution watershed delineation for up to 300 pour points
- Automated watershed boundary generation tailored to culvert locations
- Stream network extraction and analysis

### Hydrologic Vulnerability Assessment
- Rational method implementation for peak discharge estimation
- Graphical peak discharge methods
- Non-stationary regional frequency analysis
- Culvert capacity calculations for inlet and outlet control conditions
- Return period flood analysis

### Geomorphologic Risk Analysis
- RUSLE (Revised Universal Soil Loss Equation) erosion rate calculations
- Stream bank erosion vulnerability modeling
- Watershed debris flow model

### Visualization and Reporting
- Interactive map generation with multiple data layers
- Technical plot generation for reports
- Automated DOCX report creation
- Results dashboard support

## Key Features

- **Modular Design**: Independent utility functions that can be used separately or together
- **Standards-Based**: Calculations follow FHWA, USDA, and USGS methodologies
- **Geospatial Processing**: Built-in support for raster and vector data handling
- **Scalable**: Designed to handle multiple pour points and large watersheds efficiently


## Related Project

These utilities are part of the **CULVERT Tool** web application, which provides:
- Interactive web interface for vulnerability assessments
- Project management and data persistence
- Comprehensive technical report generation
- Integration with USGS data sources

Visit the main application at: **https://culvert-at-risk.org**

## Support

For questions, bug reports, or feedback:
- Email: support@culvert-at-risk.org
- Project website: https://culvert-at-risk.org
