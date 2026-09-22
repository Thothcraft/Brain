import pytest


@pytest.mark.parametrize('plan,status', [('free', 403), ('home', 403), ('research', 200)])
def test_labs_plan_gate(api, plan, status):
    client, db, user = api
    user.plan = plan
    db.commit()
    response = client.get('/api/labs')
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json() == {'labs': []}
