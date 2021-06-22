from IPython.display import display, Markdown, clear_output

from ipywidgets import Box, HBox, VBox, Layout
import ipywidgets as widgets




question_1 = {'description': '### Our independent variable is called `target`. True or False?',
              'options': ['True', 'False']}

question_2 = {'description': "### Please select the independent variable most correlated with the target.",
              'options': ['target', 'age', 'sex', 'bmi', 'bp']}

question_3 = {'description': '### Please select all features that have a `strong` correlation.',
              'options': ['age', 'sex', 'bmi', 'bp']}

question_4 = {'description': '### Please select the sentence that best interprets the intercept term.',
              'options': ['The intercept tells us that the true average for `target` is about 152.',
                          'The intercept tells us that 152 is the highest value `target` can have.',
                          'The intercept tells us that with a one unit change in the intercept, `target` will change by 152 units on average.',
                          'The intercept tells us that the average for `target` when bmi is 0, is 152.']}

question_5 = {'description': '### Please select the sentence that best interprets the slope term.',
              'options' :['The slope tells us that the average bmi is 949.', 
                         'The slope tells us that when bmi is 949, the average of `target` is  152.',
                         'The slope tells us that a 1 unit change in `bmi` results in an average increase of 949 in `target`.',
                         'The slope tells us that a 1 unit change in `bmi` results in an average increase of 152 in `target`.']}

  
    
class MultiChoice:
    
    def __init__(self, data):
        
        self.description =  data['description']
        self.options = data['options']
        self.components()
        
    def components(self):
        items_layout = Layout( width='auto')   

        self.box_layout = Layout(display='flex',
                            flex_flow='column',
                            align_items='stretch',
                            border='solid',
                            width='100%')
        
        radio_options = [(words, i) for i, words in enumerate(self.options)]
        self.alternativ = widgets.RadioButtons(
            options = radio_options,
            description = '',
            disabled = False,
            value=None,
            layout=items_layout)

        self.description_out = widgets.Output()
        with self.description_out:
            display(Markdown(self.description))

        self.feedback_out = widgets.Output()

        self.button = widgets.Button(description="submit")

class IndepedentDependent(MultiChoice):
    def __init__(self, data):
        MultiChoice.__init__(self, data)
        self.correct = 'False'
        self.button.on_click(self.check)
        
    def check(self, b):
        answer = int(self.alternativ.value)
        answer = self.options[answer]
        if answer==self.correct:
            s = "✅ Correct! We wish to *explain* the `target` column with all of the other columns in our dataset. Thus, `target` is *dependent* on all the other features and is considered the *dependent* variable. Any column we use to explain the target is considered an *independent* variable."
        else:
            s = "❌ *Incorrect. The `target` column is dependent on the other features.*"
        with self.feedback_out:
            clear_output()
            display(Markdown(s))
            
    def display(self):
        return widgets.VBox([self.description_out, self.alternativ, 
                     self.button, self.feedback_out], layout=self.box_layout)
    
    
class Correlation(MultiChoice):
    
    def __init__(self, data):
        MultiChoice.__init__(self, data)
        self.correct = 'bmi'
        self.button.on_click(self.check)


    def check(self, b):
        answer = int(self.alternativ.value)
        answer = self.options[answer]
        if answer==self.correct:
            s = "✅ **Correct!**"
        else:
            s = "❌ **Incorrect**"
        with self.feedback_out:
            clear_output()
            display(Markdown(s))
            
    def display(self):
        return widgets.VBox([self.description_out, self.alternativ, 
                     self.button, self.feedback_out], layout=self.box_layout)
    
class InterpretIntercept(MultiChoice):

    def __init__(self, data):
        MultiChoice.__init__(self, data)
        self.correct = 'The intercept tells us that the average for `target` when bmi is 0, is 152.'
        self.button.on_click(self.check)


    def check(self, b):
        answer = int(self.alternativ.value)
        answer = self.options[answer]
        if answer==self.correct:
            s = """✅ *Correct!* This is the correct interpretation of the intercept. 
            It worth acknowledging that this interpretation seems somewhat non sensical.
            *(How on earth can someone have a bmi of 0?)* This is a common issue when using linear
            regression to explain the relationships between variables. Generally speaking, even when 
            an intercept is somewhat non sensical, you still want to include it in your model. 
            The intercept helps ensure you meet certain assumptions of linear regression, and is 
            especially useful when we have categorical data."""
        elif answer == 'The intercept tells us that the true average for `target` is about 152.':
            s = """❌ *Incorrect.* The intercept tells us the expected average of the dependent variable
            *when all indepdentdent variables have a value of 0.* The idea of linear regression is that the
            dependent data doesn't have a universal average, but that instead the dependent variable's average is
            dependent on the settings of the independent variables. Generally speaking, no number from a linear regression
            model will ever be interpreted as the \"true\" average."""
        elif answer == 'The intercept tells us that 152 is the highest value `target` can have.':
            s = """❌ *Incorrect.* If the independent variables all have positive slopes, *and* the maximum value for the
            independent variables is `0` then this statement would be true. This sort of circumstance is rarely seen with 
            real world data, and thus this interpretation is largly incorrect."""
        elif answer == 'The intercept tells us that with a one unit change in the intercept, `target` will change by 152 units on average.':
            s = """❌ *Incorrect.* A one unit change in the intercept would change the intercept term to 153. 
            The intercept does not represent a slope, but rather the starting place for measuring the impact of the independent variables.
            ie, the intercept is the dependent variable's average when the independent variable has a value of 0, and if we increase the independent variable to 1, what happens to the average of the depdendent variable? That is the primary question we seek to answer when using linear regression to analyze the relationships between variables."""
        with self.feedback_out:
            clear_output()
            display(Markdown(s))
            
    def display(self):
        return widgets.VBox([self.description_out, self.alternativ, 
                     self.button, self.feedback_out], layout=self.box_layout)
    
class InterpretSlope(MultiChoice):

    def __init__(self, data):
        MultiChoice.__init__(self, data)
        self.correct = 'The slope tells us that a 1 unit change in `bmi` results in an average increase of 949 in `target`.'
        self.button.on_click(self.check)


    def check(self, b):
        answer = int(self.alternativ.value)
        answer = self.options[answer]
        if answer==self.correct:
            s = """✅ *Correct!* This is the correct interpretation of the slope. The coefficient 
            for an independent variable represents the average *effect* of that variable in terms of how it changes
            the dependent variable."""
        elif answer == 'The slope tells us that the average bmi is 949.':
            s = """❌ *Incorrect.*"""
        elif answer == 'The slope tells us that when bmi is 949, the average of `target` is  152.':
            s = """❌ *Incorrect.*"""
        elif answer == 'The slope tells us that a 1 unit change in `bmi` results in an average increase of 152 in `target`.':
            s = """❌ *Incorrect.*"""
        with self.feedback_out:
            clear_output()
            display(Markdown(s))
            
    def display(self):
        return widgets.VBox([self.description_out, self.alternativ, 
                     self.button, self.feedback_out], layout=self.box_layout)
    
    
class CheckBoxes:

    def __init__(self, data):
        
        self.description =  data['description']
        self.options = data['options']
        self.components()
        
    def components(self):
        items_layout = Layout( width='auto')   

        self.box_layout = Layout(display='flex',
                            flex_flow='column',
                            align_items='stretch',
                            border='solid',
                            width='100%')
        
        self.blocks = [widgets.Checkbox(value=False, description=option, layout=items_layout) for option in self.options]


        self.description_out = widgets.Output()
        with self.description_out:
            display(Markdown(self.description))

        self.feedback_out = widgets.Output()

        button = widgets.Button(description="submit")
        
        self.blocks.append(button)
        
class StrongCorrelation(CheckBoxes):

    def __init__(self, data):
        CheckBoxes.__init__(self, data)
        self.correct = []
        self.blocks[-1].on_click(self.check)
        
    def collect_answers(self):
        selected_data = []
        for i in range(1, len(self.blocks[1:-2])):
            print(i, self.blocks[i])
            if self.blocks[i].value == True:
                selected_data.append(self.blocks[i].description)
        self.answers = selected_data
        
    def check(self, b):
        self.collect_answers()
        
        if self.answers==self.correct:
            s = """✅ *Correct!*"""
        else:
            s = """❌ *Incorrect. Try googling 'Strong Correlation'.*"""
        with self.feedback_out:
            clear_output()
            display(Markdown(s))
            
    def display(self):
        self.blocks = [self.description_out] + self.blocks
        self.blocks.append(self.feedback_out)
        return widgets.VBox(children=self.blocks, layout=self.box_layout)
    
    
 

independent = IndepedentDependent(question_1)
correlation = Correlation(question_2)
correlation_strong = StrongCorrelation(question_3)
intercept = InterpretIntercept(question_4)
slope = InterpretSlope(question_5)

    
    
