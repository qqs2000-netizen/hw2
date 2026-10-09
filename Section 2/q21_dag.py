from datetime import datetime, timedelta
import time

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

count = 0


def sleeping_function():
    time.sleep(2)


def print_function():
    print("This represents a python operator")
    time.sleep(1)


def count_function():
    global count
    count += 1
    print("count_increase output: {}".format(count))
    time.sleep(1)


def sum_function():
    print("sum of 1..100 = {}".format(sum(range(1, 101))))


default_args = {
    "owner": "qqs2000",
    "depends_on_past": False,
    "email": ["qqs2000@columbia.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

# start_date is in the past and catchup=False, so the first run starts
# as soon as the DAG is unpaused, then it runs every 30 minutes.
with DAG(
    "q21_dag",
    default_args=default_args,
    description="HW2 Q2.1 DAG",
    schedule=timedelta(minutes=30),
    start_date=datetime(2026, 10, 8),
    catchup=False,
    max_active_runs=1,
    tags=["hw2"],
) as dag:

    # Bash operators: echo, sleep, date, run a bash script, run a python file
    t1 = BashOperator(task_id="t1", bash_command="echo 'start of the pipeline'")
    t2 = BashOperator(task_id="t2", bash_command="sleep 2")
    t3 = BashOperator(task_id="t3", bash_command="date")
    t5 = BashOperator(task_id="t5", bash_command="bash ~/airflow/scripts/hello.sh ")  # trailing space needed
    t6 = BashOperator(task_id="t6", bash_command="sleep 1")
    t9 = BashOperator(task_id="t9", bash_command="python3 ~/airflow/scripts/hello.py")
    t10 = BashOperator(task_id="t10", bash_command="echo 'running on' $(hostname)")
    t13 = BashOperator(task_id="t13", bash_command="sleep 2")
    t14 = BashOperator(task_id="t14", bash_command="date")
    t17 = BashOperator(task_id="t17", bash_command="bash ~/airflow/scripts/hello.sh ")
    t18 = BashOperator(task_id="t18", bash_command="echo 'almost done'")

    # Python operators: sleep, print, count, sum
    t4 = PythonOperator(task_id="t4", python_callable=print_function)
    t7 = PythonOperator(task_id="t7", python_callable=sleeping_function)
    t8 = PythonOperator(task_id="t8", python_callable=count_function)
    t11 = PythonOperator(task_id="t11", python_callable=sum_function)
    t12 = PythonOperator(task_id="t12", python_callable=print_function)
    t15 = PythonOperator(task_id="t15", python_callable=count_function)
    t16 = PythonOperator(task_id="t16", python_callable=sleeping_function)
    t19 = PythonOperator(task_id="t19", python_callable=sum_function)

    t1 >> [t2, t3, t4, t5]
    t2 >> t6
    t3 >> [t7, t12]
    t5 >> [t8, t9]
    t7 >> [t13, t14, t18]
    t8 >> [t10, t15]
    t9 >> [t11, t12]
    [t10, t11, t12] >> t14
    t14 >> [t16, t17]
    [t13, t15, t17] >> t18
    [t16, t18] >> t19
