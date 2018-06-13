# Jupyter Notebook 기본 경로 변경 #
출처: http://luke77.tistory.com/52 [Luke Kim]

1. Command 실행
2. jupyter notebook --generate-config 입력
3. 사용자 폴더에 .jupyter 폴더 진입 (C:\Users\Administrator\.jupyter)
4. jupyter_notebook_config.py 열기  
5. #c.NotebookApp.notebook_dir = '' 열찾기 (179 번째 line 정도)
6. 주석제거
7. '' 란 안에 원하는 폴더의 절대 경로 삽입. 단 \ --> / 로 변경 (c:\temp --> c:/temp)
8. 저장 후 jupyter notebook 재실행