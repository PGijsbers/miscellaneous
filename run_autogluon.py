from autogluon.tabular import TabularDataset, TabularPredictor
import openml

task = openml.tasks.get_task(363580)
dataset = task.get_dataset()
data, *_ = dataset.get_data()

ag_data = TabularDataset(data)
predictor = TabularPredictor(label='Class').fit(train_data=ag_data, time_limit=60)

print("Training finished successfully")
predictions = predictor.predict(data)