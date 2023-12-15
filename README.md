# SDA_Rental_prices

## HSG, Fall 2023 - Group 1 project

Welcome to our Quantlet named "**Predicive analytics for rental prices using diverse machine learning methods: Zurich case study**"

The project intented to extract and clean web data from the homelisting website Immobilier.ch for the region of Zurich. In addition, shapefiles from open source Zurich Stadt that allows to leverage spatial data and create more covariate to perform predicive analysis.

Please follow those instruction to reproduce our results:

0) Download the files on your machines
   Make sure to extract zip files. In particular, merge the content of "ZH canton part 1" and "ZH canton part 2" into a single folder called "ZH canton"
   Download the shapefile called "Gebäudealter.shp" on the following page: https://data.stadt-zuerich.ch/dataset/ktzh_gebaeudealter__ogd_ and save it in folder "ZH canton" as well.

2) Run Immobilier.ch_scrapping_Selenium.py
   Keep in mind that Selenium takes quie some time to run depending on the number of pages to scrap.
   Make sure to remove the comment mark (#) on lines 116, 317 and 366 to save your datasets.

3) Run Spatial_analysis_ZH.py to load shapefiles and create new covariates
   Keep in mind that Nominatim API doesn't allow to perform more than 1 request per second (don't touch the time.sleep(1))
   Make sure to remove the comment mark (#) on lines 40, 119 and 142 to save your datasets.

4) Run Final_Group_1

Feel free to consulte quantinar.com to learn more about the technics used.
Have fun !
