from autogluon.tabular import TabularDataset, TabularPredictor
import openml
import sys

_, time, task = sys.argv

print("Downloading OpenML task")
task = openml.tasks.get_task(int(task))
dataset = task.get_dataset()
data, *_ = dataset.get_data()

print("Training AutoGluon")
ag_data = TabularDataset(data)
predictor = TabularPredictor(label='Class').fit(train_data=ag_data, time_limit=int(time))

print("Training finished successfully")
predictions = predictor.predict(data)