import pickle as pkl
from pathlib import Path

with Path.open("bench_data.pkl", "rb") as file:
    data_bench = pkl.load(file)
    x_list_bench = data_bench[0]
    u_list_bench = data_bench[1]
    xref_list_bench = data_bench[2]

with Path.open("gym_data.pkl", "rb") as file:
    data_gym = pkl.load(file)
    x_list_gym = data_gym[0]
    u_list_gym = data_gym[1]
    xref_list_gym = data_gym[2]

for i in range(len(u_list_bench)):
    u_bench = u_list_bench[i]
    u_gym = u_list_gym[i]

    if not (u_bench == u_gym).all():
        print("different torque at index: " + str(i))
        break

for i in range(len(u_list_bench)):
    x_bench = x_list_bench[i]
    x_gym = x_list_gym[i]

    if not (x_bench == x_gym).all():
        print("differente state at index: " + str(i))
        break

print(len(xref_list_bench))
print(len(xref_list_gym))

for i in range(len(xref_list_bench)):
    xref_bench = xref_list_bench[i]
    xref_gym = xref_list_gym[i]

    if not (xref_bench == xref_gym).all():
        print("differente reference at index: " + str(i))
        break
print("test done")
