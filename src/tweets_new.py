from twarc import Twarc2, expansions
import datetime
import numpy as np
import pandas as pd
import json
import io
import csv
from googletrans import Translator
import time
translator = Translator()

# Replace your bearer token below
client = Twarc2(bearer_token="AAAAAAAAAAAAAAAAAAAAAPt5NwEAAAAAMIHNOlGJu1WJvosdeQCxfKEC6jU%3Dc4Up13YR0pZAvZGJTN5M0y49VvnbMCbrOqyEMIAyyRnxliUvs7")


def main():
   
    tweets = []
    for line in open('tweets.txt', 'r'):
        tweets.append(json.loads(line))
    
               




if __name__ == "__main__":
    main()