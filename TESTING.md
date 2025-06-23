## Test Evaluation Framework

The project includes a comprehensive test evaluation framework to assess the performance and accuracy of the HealHub application. This framework is designed to evaluate all major components systematically and provide detailed metrics for continuous improvement.

### Framework Components

1. **Test Suite Structure**
   - `tests/test_evaluation.py`: Main evaluation framework
   - `tests/test_data/`: Directory containing test cases
     - `nlu_test_cases.json`: NLU evaluation test cases
     - `symptom_checker_test_cases.json`: Symptom checker evaluation test cases
     - `response_generator_test_cases.json`: Response generator evaluation test cases

2. **Evaluation Metrics**

   a. **NLU Metrics**:
   - Intent accuracy
   - Entity recall and precision
   - Emergency detection accuracy
   - Language detection accuracy

   b. **Symptom Checker Metrics**:
   - Symptom identification accuracy
   - Follow-up question relevance
   - Assessment quality
   - Triage accuracy

   c. **Response Generator Metrics**:
   - Response relevance
   - Safety compliance
   - Disclaimer inclusion

3. **Test Cases**
   - Multi-language support (10 Indian languages)
   - Various query types (symptoms, general health, emergencies)
   - Different severity levels
   - Complex symptom combinations
   - Safety and disclaimer requirements

### Running the Evaluation

1. **Setup**:
   ```bash
   # Ensure you have the required dependencies
   pip install -r requirements.txt

   # Set up your API key
   export SARVAM_API_KEY="your_api_key_here"
   ```

   > **Important Note**: The evaluation framework uses actual Sarvam AI API calls to assess performance. This means:
   > - Each test case will consume your API quota
   > - Results reflect real API performance
   > - Tests may take longer to run due to API latency
   > - You need a valid API key with sufficient quota
   > - Consider running tests during off-peak hours to minimize API costs

2. **Run Tests**:
   ```bash
   # Run all evaluation tests
   python -m unittest tests/test_evaluation.py -v

   # Generate evaluation report
   python -c "
   from tests.test_evaluation import HealHubEvaluator
   evaluator = HealHubEvaluator(api_key='your_api_key_here')
   evaluator.save_evaluation_results('evaluation_results.txt')
   "
   ```

3. **Review Results**:
   - Check `evaluation_results.txt` for a human-readable report
   - Review `evaluation_results_metrics.json` for detailed numerical data
   - Use the metrics to identify areas for improvement

### Adding New Test Cases

1. **NLU Test Cases**:
   ```json
   {
     "input_text": "Your test query",
     "language": "language-code",
     "expected_intent": "INTENT_TYPE",
     "expected_entities": [
       {"text": "entity", "type": "entity_type"}
     ],
     "is_emergency": false
   }
   ```

2. **Symptom Checker Test Cases**:
   ```json
   {
     "input_text": "Symptom description",
     "language": "language-code",
     "symptoms": ["symptom1", "symptom2"],
     "expected_symptoms": ["expected1", "expected2"],
     "expected_questions": ["question1", "question2"],
     "simulated_answers": {
       "symptom1": {"question1": "answer1"}
     },
     "expected_assessment": {
       "suggested_severity": "severity_level",
       "key_points": ["point1", "point2"]
     }
   }
   ```

3. **Response Generator Test Cases**:
   ```json
   {
     "input_text": "User query",
     "nlu_result": {
       "intent": "QUERY_TYPE",
       "entities": [{"text": "entity", "type": "type"}]
     },
     "expected_response": {
       "key_points": ["point1", "point2"]
     },
     "safety_requirements": {
       "required_phrases": ["phrase1", "phrase2"]
     }
   }
   ```

### Continuous Integration

The evaluation framework can be integrated into CI/CD pipelines to:
- Run automated tests on each commit
- Track performance metrics over time
- Generate reports for each test run
- Alert on significant performance changes

### Best Practices

1. **Test Case Management**:
   - Keep test cases organized by component
   - Include both positive and negative test cases
   - Cover edge cases and error scenarios
   - Maintain test data in JSON format for easy updates

2. **Metrics Tracking**:
   - Monitor metrics over time
   - Set baseline performance expectations
   - Track improvements after changes
   - Document significant changes in performance

3. **Reporting**:
   - Generate regular evaluation reports
   - Include both summary and detailed metrics
   - Highlight areas needing improvement
   - Track progress over time

4. **Maintenance**:
   - Regularly update test cases
   - Add new test scenarios as features are added
   - Remove obsolete test cases
   - Keep documentation up to date
