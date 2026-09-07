def iterate_dataset(
    groups,
    subjects,
    tasks,
    stages,
    blocks=None,
    adaptation_splits=None
):
    """
    Iterate through all group-subject-task-stage combinations.

    Yields:
        group, subject, task, stage
    """

    for group in groups:

        for subject in subjects[group]:

            for task in tasks:

                for stage in stages:
                    if blocks is not None and task == "DeCRAT":
                        for block in blocks:
                            if adaptation_splits and block == "adaptation":
                                for adaptation_split in adaptation_splits:
                                    yield (
                                        group,
                                        subject,
                                        task,
                                        stage,
                                        block,
                                        adaptation_split
                                    )
                            else:
                                yield (
                                    group,
                                    subject,
                                    task,
                                    stage,
                                    block,
                                    None
                                )
                    
                    else:
                        yield (
                            group,
                            subject,
                            task,
                            stage,
                            None,
                            None
                        )
