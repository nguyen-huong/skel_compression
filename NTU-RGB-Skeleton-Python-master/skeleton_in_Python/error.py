import csv
import pandas as pd

def readFile(fileName = "skeleton_data_error.csv"):
    fileIn = open(fileName, "r")
    readData = csv.reader(fileIn)
    data_head = next(readData)
    print (data_head)

    #pandas
    table_data = pd.read_csv( "skeleton_data_error.csv")
    print(table_data)
    min = table_data['error'].min()
    max = table_data['error'].max()
    avg = table_data['error'].mean()
    print("The lowest error value: ", min)
    print("The highest error value: ", max)
    print("Mean: ", avg)

    fileIn.close()


readFile(fileName = "skeleton_data_error.csv")