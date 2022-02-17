Student ID: 210049506
Name: RAJANI MOHAN JANIPALLI

INM431 Machine Learning Coursework, City University of London

Notes of running the Code Scripts.


	SOFTWARE VERSIONS USED:
1. MATLAB
Version: R2021a Update 5 (9.10.0.1739362)
	 64-bit (Win64)
	 August 9, 2021

2. JUPYTER NOTEBOOK
Version: 6.4.5
	 Anaconda 3

**NOTE : Since the code scripts have been prepared in the above
	 software versions, it is recommended to run them on the
	 same versions for a smooth experience.


	Details of files in the zip folder:
  File Name/type	  	Description
1.compas/.csv 		  	Original data set taken from OpenML website.
2.compas_Desc_Stats/.csv  	File created in python after second stage of EDA in python which
			  	contains just a table descriptive statistics of the data.
3.compas_Desc_Stats/.xlsx 	Excel version of the above file that was made to add colors and
			  	style, without any changes to the contents, so that a screenshot
			  	of the table in the file can be put as a figure in the poster.
4.compas_for_EDA/.csv	  	File created in MATLAB after first stage of EDA and to be used in
			  	python for second stage of EDA.
5.Recidivismtestset/.csv  	File containing test set, created in MATLAB after EDA and partition
			  	of original data set.
6.Recidivismtrainset/.csv 	File containing train set, created in MATLAB after EDA and partition
			  	of original data set.
7.RecidivisimDataEDA/.m	  	MATLAB script of first stage of EDA and partition of original data 
				set into train and test sets.
8.Recidivism EDA Continuation	Jupyter Notebook containing second stage of EDA.
  /.ipynb
9.RecidivismLogisticRegression	MATLAB script of Logistic Regression model training and testing.
  /.m
10.RecidivismRandomForest	MATLAB script of Random Forest model training and testing.
  /.m


**NOTE for execition of Codes - To understand the smooth flow of code, it is recommended to
				run each section of the scripts one by one and not run the whole
				script in a single go.


	Order of running the codes:
1.Before running the codes it is requested to keep all the files in the zip folder intact
  and together, till the assesment is over.

2.First please open the file "RecidivisimDataEDA.m" and run all the sections one by one. After the
  first stage of EDA is over in this script, comments in the script will indicative the end 
  of first stage of EDA and then you may to proceed for the second stage of the EDA in
  python. But please keep this file opened.

3.Then open the file "Recidivism EDA Continuation.ipynb" and run each cell of the script one by one.
  At the end of the script comments will indicate the end of second stage of EDA. Then you may
  close this jupyter notebook and go back to the "RecidivisimDataEDA.m" file.

4. After the EDA, the data set has been partitioned into training and test set which can be checked
   in the codes further in the script. After reaching the end of the script please close this file.

5.Then open the file "RecidivismLogisticRegression.m" and run each section of the script one by one
  till the end. Then please close this file.

6.Then open the file "RecidivismRandomForest.m" and run each section of the script one by one
  till the end. Then please close this file.

7. That's it.

			^^^ THANK YOU FOR YOUR TIME AND PATIENCE ^^^