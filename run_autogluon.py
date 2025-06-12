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

train_ind, test_ind = task.get_train_test_split_indices()
train_data = TabularDataset(data.iloc[train_ind,:])
test_data = TabularDataset(data.iloc[test_ind,:])

print("Training AutoGluon")
predictor = TabularPredictor(
    label='Class', verbosity=4
).fit(
    train_data=train_data,
    test_data=test_data,
    time_limit=int(time)
)

print("Training finished successfully")
predictions = predictor.predict(data)