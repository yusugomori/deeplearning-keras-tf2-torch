class Company:
    def __init__(self, sales, cost, persons):
        self.sales = sales
        self.cost = cost
        self.persons = persons

    def get_profit(self):
        return self.sales - self.cost


company_A = Company(100, 80, 10)
company_B = Company(40, 60, 20)

print(company_A.sales)
print(company_A.get_profit())

company_A.sales = 80
print(company_A.sales)
