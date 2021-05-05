import tomotopy as tp
import pandas as pd

from lda3 import preprocess

domain_stopwords = ["ames","armstrong","glenn","goddard","katherine","johnson",
                    "jpl","kennedy","langley","marshall","michoud","nasa","plum",
                    "brook","stennis","wallops","white","sands","sand","sandusky",
                    "ohio","cleveland","fairmont","west","virginia","greenbelt",
                    "maryland","virginia","washington","hampton","florida",
                    "huntsville","alabama","mississippi","orleans","louisiana",
                    "houston","texas","las","cruces","mexico","dryden","edwards",
                    "california","pasadena","moffett","field","york","albuquerque",
                    "robert","moscow","sofia", "january", "february", "march", 
                    "april", "may","june", "july", "august", "september", "october", 
                    "november", "december", "csoc", "gsfc", "cofr", "keyword",
                    "germany", "russia", "inch", "meter", "usml", "morning",
                    "also", "lesson", "learn", "lockheed", "martin", "northrop", "grumman",
                    "determine", "pick", "canoga", "park", "william", "think", "please",
                    "refer", "totally", "month", "day", "year", "vandenberg", "senior",
                    "msfc","cause"]
list_of_attributes=['Lesson(s) Learned','Driving Event','Recommendation(s)']

def get_data(csv_file_name):
    def load_data(csv_file_name):
        df = pd.read_csv(open(csv_file_name,encoding='utf8',errors='ignore'))
        return df
    def remove_incomplete_rows(df):
        columns_to_drop = ['Submitter 1', 'Submitter 2', 'Submitter 3', 'Submitter 4',
                               'Submitter 5', 'Pont of Contact 1','Pont of Contact 2',
                               'Pont of Contact 3','Pont of Contact 4','Pont of Contact 5',
                               'Contributor 1','Contributor 2','Contributor 3','Contributor 4',
                               'Contributor 5','Organization', 'Abstract',
                               'Date Lesson Occurred','Evidence','Project / Program',
                               'The related NASA policy(s), standard(s), handbook(s), procedure(s) or other rules',	
                               'NASA Mission Directorate(s)','Sensitivity',
                               'From what phase of the program or project was this lesson learned captured?',
                              'Where (other lessons, presentations, publications, etc.)?',
                              'Publish Date','Topics']
        rows_to_drop = []
        for i in range (0, len(df)):
            if str(df.iloc[i]["Recommendation(s)"]).strip("()").lower().startswith("see") or str(df.iloc[i]["Recommendation(s)"]).strip("()").lower().startswith("same") or str(df.iloc[i]["Recommendation(s)"])=="" or isinstance(df.iloc[i]["Recommendation(s)"],float) or str(df.iloc[i]["Recommendation(s)"]).lower().startswith("none"):
                rows_to_drop.append(i); continue
            if str(df.iloc[i]['Driving Event']).strip("()").lower().startswith("see") or str(df.iloc[i]['Driving Event']).strip("()").lower().startswith("same") or str(df.iloc[i]['Driving Event'])=="" or isinstance(df.iloc[i]['Driving Event'], float) or str(df.iloc[i]['Driving Event']).lower().startswith("none"):
                 rows_to_drop.append(i); continue
            if str(df.iloc[i]["Lesson(s) Learned"]).strip("()").lower().startswith("see") or str(df.iloc[i]["Lesson(s) Learned"]).strip("()").lower().startswith("same") or str(df.iloc[i]["Lesson(s) Learned"])=="" or isinstance(df.iloc[i]["Lesson(s) Learned"], float) or str(df.iloc[i]["Lesson(s) Learned"]).lower().startswith("none"):
                 rows_to_drop.append(i); continue
        df = df.drop(columns_to_drop, axis=1)
        df = df.drop(rows_to_drop, axis=0)
        df = df.reset_index()
        return df
    df = load_data(csv_file_name)
    df = remove_incomplete_rows(df)
    return df #,lesson_numbers

csv_file_name = "useable_LL_combined.csv"

data = get_data(csv_file_name)
data = preprocess(data,domain_stopwords,list_of_attributes=list_of_attributes)

lesson_numbers = data['Lesson ID']

mdl = tp.HLDAModel.load('Recommendation(s)hlda_model_object-SAVE.bin')

doc_num = 0
for doc in mdl.docs:
    print(doc_num)
    topic_nums = doc.path
    words = ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[1])])
    print(words)
    words = ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[2])])
    print(words)
    doc_num = doc_num+1
print(len(lesson_numbers))