import numpy as np

Admitted = 'Admitted'
Rejected = 'Rejected'
Male = 'Male'
Female = 'Female'
Departments = ('A', 'B', 'C', 'D', 'E', 'F')
A, B, C, D, E, F = Departments

#(Admit,Gender,Dept,Freq),
data = np.array([
        (Admitted, Male, A, 512),
        (Rejected, Male, A, 313),
        (Admitted, Female, A, 89),
        (Rejected, Female, A, 19),
        (Admitted, Male, B, 353),
        (Rejected, Male, B, 207),
        (Admitted, Female, B, 17),
        (Rejected, Female, B, 8),
        (Admitted, Male, C, 120),
        (Rejected, Male, C, 205),
        (Admitted, Female, C, 202),
        (Rejected, Female, C, 391),
        (Admitted, Male, D, 138),
        (Rejected, Male, D, 279),
        (Admitted, Female, D, 131),
        (Rejected, Female, D, 244),
        (Admitted, Male, E, 53),
        (Rejected, Male, E, 138),
        (Admitted, Female, E, 94),
        (Rejected, Female, E, 299),
        (Admitted, Male, F, 22),
        (Rejected, Male, F, 351),
        (Admitted, Female, F, 24),
        (Rejected, Female, F, 317)],
        dtype=[('a', 'S15'), ('b', 'S15'), ('c', 'S15'), ('d', '<i2')])

for typ in [Male, Female]:
    gender = data[data['b'] == typ]
    for dep in Departments:
        dep = gender[gender['c']  == dep]

        adm = dep[dep['a'] == Admitted]['d'][0] * 1.0
        rej = dep[dep['a'] == Rejected]['d'][0] * 1.0
        #print adm, rej

        total = adm + rej

        print "Admitted %s %.3f" % (typ , (adm / total))
        
