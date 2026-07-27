def iterate_dataset(
    groups,
    subjects,
    tasks,
    stages
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

                    yield (
                        group,
                        subject,
                        task,
                        stage
                    )