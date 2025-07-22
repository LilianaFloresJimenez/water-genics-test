## Troubleshooting:
- In my case I had issues to setup a virtual env. The issue was that my python was a pyenv python. I needed to make sure that pyenv is disabled for this project.
- Sometimes the mflow ui shows "Forbidden" messages. Then a browser reload without cache helps.
- The mocking in tests seems to work unreliable across different operation systems. On Mac Os it helped to mock also
`patch('mlflow.sklearn') as mock_sklearn`