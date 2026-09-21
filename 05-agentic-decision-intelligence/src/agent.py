from dataclasses import dataclass,field

@dataclass
class Tool:
    name:str
    fn:object

@dataclass
class Agent:
    tools:dict=field(default_factory=dict)
    memory:list=field(default_factory=list)
    audit:list=field(default_factory=list)
    def register(self,tool): self.tools[tool.name]=tool
    def act(self,name,**kwargs):
        if name not in self.tools: raise ValueError("Unknown tool")
        result=self.tools[name].fn(**kwargs)
        self.audit.append({"tool":name,"input":kwargs,"output":result})
        return result

def risk_check(value,threshold):
    return {"decision":"accept" if value>=threshold else "review"}

if __name__=="__main__":
    a=Agent(); a.register(Tool("risk_check",risk_check))
    print(a.act("risk_check",value=.72,threshold=.65))
