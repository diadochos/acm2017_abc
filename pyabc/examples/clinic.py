import numpy as np
from .example import Example

def closing_time(y):
    return np.atleast_1d(y[2])

def nr_patients(y):
    return np.atleast_1d(y[0])

def nr_waiting_patients(y):
    return np.atleast_1d(np.sum(np.where(y[1] > 0, 1, 0)))

class Clinic(Example):

    def simulator(self, lmbd):
        lmbd = abs(lmbd)
        nr_doctors = 3
        opening_time = 9 * 60
        closing_time = 16 * 60
        patients_in_treatment = np.zeros(nr_doctors) # 0,1 first column, treatment time second column

        result = [
            [0], # nr_patients
            [], # waiting_times
            [0] # closing time
        ]

        def treatment_time():
            return np.random.uniform(5,20)

        def next_patient(last):
            return last + np.random.exponential(1/lmbd)

        def create_patient_list():
            list_of_patients = []
            list_of_patients.append(next_patient(opening_time))
            while list_of_patients[-1] < closing_time:
                list_of_patients.append(next_patient(list_of_patients[-1]))

            return np.array(list_of_patients[:-1])

        list_of_patients = create_patient_list()
        N = list_of_patients.shape[0]
        result[0][0] = N
        result[1] = np.zeros(N)

        for i, arrival in enumerate(list_of_patients):

            doctor = np.argmin(patients_in_treatment)
            treatment_end = patients_in_treatment[doctor]
            # is any doctor free? T
            # This is the case when treatment time is 0 or treatment is done before next patient arrives
            if treatment_end == 0 or arrival > treatment_end:
                patients_in_treatment[doctor] = arrival + treatment_time() # treamtment lasts between 5 and 20 minutes

            else:
                waiting_time = treatment_end - arrival # patient has to wait
                result[1][i] = waiting_time
                patients_in_treatment[doctor] = treatment_end + treatment_time() # treatment lasts between 5 and 20 minutes

        result[2][0] = np.max(patients_in_treatment)

        return result

    def _summaries(self):
        return [closing_time, nr_patients, nr_waiting_patients]

clinic = Clinic()
