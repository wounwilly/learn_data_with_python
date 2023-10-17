import sys
import os

tables = {
  1: ['Jiho', False],
  2: [],
  3: [],
  4: [],
  5: [],
  6: [],
  7: [],
}
print("tables:", tables)

def assign_table(table_number, name, vip_status=False):
  list_nameStatus = None

  if list_nameStatus is None:
    list_nameStatus = []
    list_nameStatus.append(name)
    list_nameStatus.append(vip_status)

  if tables[table_number] == []:
        tables[table_number] = list_nameStatus
  else:
        tables.update({table_number: list_nameStatus})

  print("tables:", tables)
print("--------\n")

 
assign_table(1, "woun", False)