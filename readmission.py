if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    mimic3_ds = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )

    mimic3_ds.stat()

    mimic3_ds.info()

    from pyhealth.tasks import readmission_prediction_mimic3_fn

    mimic3_ds = mimic3_ds.set_task(task_fn=readmission_prediction_mimic3_fn)
    # stats info
    mimic3_ds.stat()

    from pyhealth.datasets.splitter import split_by_patient
    from pyhealth.datasets import split_by_patient, get_dataloader

    # data split
    train_dataset, val_dataset, test_dataset = split_by_patient(mimic3_ds, [0.8, 0.1, 0.1])

    # create dataloaders (they are <torch.data.DataLoader> object)
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    from pyhealth.models import Transformer

    model = Transformer(
        dataset=mimic3_ds,
        # look up what are available for "feature_keys" and "label_keys" in dataset.samples[0]
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="binary",
    )

    from pyhealth.trainer import Trainer

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="pr_auc",
    )

    # option 1: use our built-in evaluation metric
    score = trainer.evaluate(test_loader)
    print(score)

    # option 2: use our pyhealth.metrics to evaluate
    from pyhealth.metrics.binary import binary_metrics_fn

    y_true, y_prob, loss = trainer.inference(test_loader)
    binary_metrics_fn(y_true, y_prob, metrics=["pr_auc"])