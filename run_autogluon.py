from autogluon.tabular import TabularDataset, TabularPredictor
import openml
import sys

if len(sys.argv) == 3:
    _, time, task = sys.argv
else:
    time, task = 60, 363580

print("Downloading OpenML task")
task = openml.tasks.get_task(int(task))
dataset = task.get_dataset()
data, *_ = dataset.get_data()

print("Training AutoGluon")
ag_data = TabularDataset(data)
predictor = TabularPredictor(label='Class', verbosity=4).fit(train_data=ag_data, time_limit=int(time))

print("Training finished successfully")
predictions = predictor.predict(data)