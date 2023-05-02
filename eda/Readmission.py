#!/usr/bin/env python
# coding: utf-8

# # 

# ## packages installation 

# In[1]:


# get_ipython().system('pip install pyhealth')


# ## Load dataset

# description on each table here:
# 
# 1. DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/
# 
# ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
# SUBJECT_ID is unique to a patient and HADM_ID is unique to a patient hospital stay.
# SEQ_NUM provides the order in which the ICD diagnoses relate to the patient
# ICD9_CODE contains the actual code corresponding to the diagnosis assigned to the patient for the given row
# 
# Links to:
# 
# PATIENTS on SUBJECT_ID
# ADMISSIONS on HADM_ID
# D_ICD_DIAGNOSES on ICD9_CODE
# 
# 2. PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/
# 
# ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
# 
# ICD9_CODE provides the ICD-9 code for the given procedure
# 
# 3. PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/
# 
# ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, DRUG
# 
# Links to:
# 
# PATIENTS on SUBJECT_ID
# ADMISSIONS on HADM_ID
# ICUSTAYS on ICUSTAY_ID

# In[18]:


from pyhealth.datasets import MIMIC3Dataset

mimic3_ds = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
)

mimic3_ds.stat()

# In[3]:


# data format
mimic3_ds.info()

# In[5]:


mimic3_ds.patients['1']

# In[8]:


mimic3_ds.tables

# In[17]:


# mimic3_ds.samples[0]

# In[ ]:


# ## Define healthcare task

# In[9]:


from pyhealth.tasks import readmission_prediction_mimic3_fn

mimic3_ds = mimic3_ds.set_task(task_fn=readmission_prediction_mimic3_fn)
# stats info
mimic3_ds.stat()

# In[10]:


from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(mimic3_ds, [0.8, 0.1, 0.1])

# create dataloaders (they are <torch.data.DataLoader> object)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

# ## Define ML Model

# In[11]:


from pyhealth.models import Transformer

model = Transformer(
    dataset=mimic3_ds,
    # look up what are available for "feature_keys" and "label_keys" in dataset.samples[0]
    feature_keys=["conditions", "procedures"],
    label_key="label",
    mode="binary",
)

# ## training

# In[12]:


from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=3,
    monitor="pr_auc",
)

# ##  Evaluation

# In[13]:


# option 1: use our built-in evaluation metric
score = trainer.evaluate(test_loader)
print(score)

# option 2: use our pyhealth.metrics to evaluate
from pyhealth.metrics.binary import binary_metrics_fn

y_true, y_prob, loss = trainer.inference(test_loader)
binary_metrics_fn(y_true, y_prob, metrics=["pr_auc"])

# In[ ]:
