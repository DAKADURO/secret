from motor_rag import MotorRAG
motor = MotorRAG()
query_es = 'que servicio y que elementos se remplazan a las 4000 horas'
query_en = '4000 hours maintenance replacement'
print('--- ES ---')
res_es = motor.db.similarity_search(query_es, k=3, filter={'': [{'marca': 'KAISHAN'}, {'modelo': 'KRSD-125'}]})
for r in res_es: print(r.page_content[:150].replace('\n', ' '))
print('--- EN ---')
res_en = motor.db.similarity_search(query_en, k=3, filter={'': [{'marca': 'KAISHAN'}, {'modelo': 'KRSD-125'}]})
for r in res_en: print(r.page_content[:150].replace('\n', ' '))
