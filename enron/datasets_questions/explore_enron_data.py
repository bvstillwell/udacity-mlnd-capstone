#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)
print len( enron_data.items()[0][1])


poi = [key for key, val in enron_data.iteritems() if val["poi"]]

keys = [key for key, val in enron_data.iteritems()]
print "\n".join(sorted(keys))
print enron_data["PRENTICE JAMES"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

for person in ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"]:
	print person, enron_data[person]["total_payments"]

people = [key for key, val in enron_data.iteritems() if val["total_payments"] == "NaN"]
print people
print len(people) / (1.0 * len(enron_data))

people = [key for key, val in enron_data.iteritems() if val["total_payments"] == "NaN" and val["poi"]]
print people
print len(people) / (1.0 * len(enron_data))