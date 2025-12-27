Dataset authors: Mateusz Erezman, Michał Hajdasz, Dariusz Kobiela, Adam Kurowski, Szymon Zaporowski
In order to make reduce class imbalance dataset can be combined with part of dataset published in paper 
*"An auditory dataset of passing vehicles recorded with a smartphone. written by Bazilinskyy, Pavlo & Aa, Arne & Schoustra, Michael & Spruit, John & Staats, Laurens & van der Vlist, Klaas Jan & de Winter, Joost. (2018)"*
which is publicly available at http://doi.org/10.4121/uuid:bef54ab8-73ef-42f3-b6b7-54e011737e72
To use it download addtional dataset and paste files to .\data\additional\car and .data\additional\motorcycle directories.
Then replace instead of using labels.csv use labels_with_additional_data.csv

labels.csv file consists two columns:
- file_path - relevent path to the file with sample
- class - number equal to 0, 1 or 2 meaning:
	- 0 - car class
	- 1 - truck, bus and van class
	- 2 - motorcycle class