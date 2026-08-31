def iterate_dataset(
    groups,
    subjects,
    tasks,
    stages,
    blocks=None
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
                            yield (
                                group,
                                subject,
                                task,
                                stage,
                                block
                            )
                    
                    else:
                        yield (
                            group,
                            subject,
                            task,
                            stage
                        )
