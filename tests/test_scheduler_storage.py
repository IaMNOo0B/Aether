from aether.storage import Store


def test_due_schedule_atomically_enqueues_once_and_honors_max_runs(tmp_path):
    store = Store(tmp_path / "scheduler.sqlite3")
    store.create_schedule(
        "schedule-1", "Research a public technical topic",
        interval_seconds=10, max_runs=1, now_epoch=100,
    )

    assert store.enqueue_due_schedules(now_epoch=109) == []
    created = store.enqueue_due_schedules(now_epoch=110)
    assert len(created) == 1
    assert store.enqueue_due_schedules(now_epoch=111) == []

    task = store.get_task(created[0])
    assert task["status"] == "queued"
    assert task["state"]["scheduled"] is True
    assert task["state"]["scheduled_goal_id"] == "schedule-1"
    assert store.task_events(created[0])[0]["event_type"] == "scheduled_task_created"

    schedule = store.list_schedules()[0]
    assert schedule["run_count"] == 1
    assert schedule["enabled"] == 0


def test_schedule_pause_prevents_due_task_creation(tmp_path):
    store = Store(tmp_path / "scheduler.sqlite3")
    store.create_schedule(
        "schedule-2", "Check a public source", interval_seconds=10, now_epoch=100
    )
    store.set_schedule_enabled("schedule-2", False)
    assert store.enqueue_due_schedules(now_epoch=1000) == []
