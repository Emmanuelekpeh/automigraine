"""
Sequential agent orchestration for AutoMigraine
Agents work in a predetermined sequence with each building on previous work
"""

import asyncio
from typing import Dict, List, Any

class SequentialOrchestrator:
    """Orchestrates multiple agents working sequentially"""
    
    def __init__(self, model_connectors: Dict[str, Any]):
        """Initialize with model connectors for each agent role"""
        self.connectors = model_connectors
    
    async def process_task(self, task: str) -> Dict[str, Any]:
        """Process a task using sequential agents"""
        results = {
            "task": task,
            "steps": []
        }
        
        # Step 1: Manager analyzes and creates a plan
        plan_prompt = f"""You are the Manager agent. Create a detailed plan for this task:

Task: {task}

Break this down into sequential steps that must be completed in order.
For each step, specify:
1. What needs to be done
2. What information is needed from previous steps
3. Expected output

Format as a numbered list with clear instructions.
"""
        plan = await self.connectors["manager"].generate(plan_prompt)
        results["steps"].append({
            "role": "manager",
            "action": "planning",
            "output": plan
        })
        
        # Step 2: Developer 1 creates an initial implementation
        dev1_prompt = f"""You are Developer 1. Create an initial implementation based on this plan:

Task: {task}

Manager's Plan:
{plan}

Provide a complete implementation that follows the plan.
Include code, documentation, and explanations as needed.
"""
        dev1_solution = await self.connectors["developer1"].generate(dev1_prompt)
        results["steps"].append({
            "role": "developer1",
            "action": "implementation",
            "output": dev1_solution
        })
        
        # Step 3: Developer 2 reviews and enhances the implementation
        dev2_prompt = f"""You are Developer 2. Review and enhance this implementation:

Task: {task}

Manager's Plan:
{plan}

Developer 1's Implementation:
{dev1_solution}

Your job is to:
1. Identify any issues or gaps in the implementation
2. Suggest improvements or optimizations
3. Add any missing features
4. Ensure the solution fully addresses the original task

Provide your enhanced version of the solution.
"""
        dev2_solution = await self.connectors["developer2"].generate(dev2_prompt)
        results["steps"].append({
            "role": "developer2",
            "action": "review_and_enhance",
            "output": dev2_solution
        })
        
        # Step 4: Moderator finalizes and polishes the solution
        moderator_prompt = f"""You are the Moderator. Finalize and polish this solution:

Task: {task}

Manager's Plan:
{plan}

Developer 1's Implementation:
{dev1_solution}

Developer 2's Enhanced Version:
{dev2_solution}

Your job is to:
1. Resolve any conflicts or inconsistencies
2. Ensure the solution is complete and correct
3. Format the solution for clarity and readability
4. Provide a final assessment of quality

Deliver a final, polished version of the solution.
"""
        final_solution = await self.connectors["moderator"].generate(moderator_prompt)
        results["steps"].append({
            "role": "moderator",
            "action": "finalize",
            "output": final_solution
        })
        
        # Extract the final result
        results["final_solution"] = final_solution
        
        return results
    
    async def process_task_with_feedback(self, task: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Process a task with feedback loops between agents"""
        results = {
            "task": task,
            "iterations": []
        }
        
        current_solution = None
        
        for i in range(max_iterations):
            iteration_results = {
                "iteration": i + 1,
                "steps": []
            }
            
            # Step 1: Manager creates/updates plan based on current state
            plan_prompt = f"""You are the Manager agent. {'Create' if i==0 else 'Update'} the plan for this task:

Task: {task}

{'Here is the current solution:' if current_solution else ''}
{current_solution if current_solution else ''}

{'Identify what issues need to be addressed in this iteration.' if current_solution else 'Create an initial plan for implementing this task.'}
Break down the work into clear steps.
"""
            plan = await self.connectors["manager"].generate(plan_prompt)
            iteration_results["steps"].append({
                "role": "manager",
                "action": "planning",
                "output": plan
            })
            
            # Step 2: Developer 1 implements based on plan and previous solution
            dev1_prompt = f"""You are Developer 1. Implement the solution based on:

Task: {task}

Current Plan:
{plan}

{'Previous Solution:' if current_solution else ''}
{current_solution if current_solution else ''}

Provide {'an improved' if current_solution else 'a complete'} implementation.
"""
            dev1_solution = await self.connectors["developer1"].generate(dev1_prompt)
            iteration_results["steps"].append({
                "role": "developer1", 
                "action": "implementation",
                "output": dev1_solution
            })
            
            # Step 3: Developer 2 reviews and enhances
            dev2_prompt = f"""You are Developer 2. Review and enhance:

Task: {task}

Current Plan:
{plan}

Developer 1's Implementation:
{dev1_solution}

Provide specific improvements and enhancements.
"""
            dev2_solution = await self.connectors["developer2"].generate(dev2_prompt)
            iteration_results["steps"].append({
                "role": "developer2",
                "action": "review_and_enhance",
                "output": dev2_solution
            })
            
            # Step 4: Moderator evaluates and integrates
            moderator_prompt = f"""You are the Moderator. Evaluate and integrate solutions:

Task: {task}

Current Plan:
{plan}

Developer 1's Implementation:
{dev1_solution}

Developer 2's Enhancements:
{dev2_solution}

Provide a cohesive integrated solution and evaluate if further iterations are needed.
If this is iteration {i+1} of {max_iterations}, determine if the solution is complete.
"""
            moderator_solution = await self.connectors["moderator"].generate(moderator_prompt)
            iteration_results["steps"].append({
                "role": "moderator",
                "action": "integrate_and_evaluate",
                "output": moderator_solution
            })
            
            # Update current solution for next iteration
            current_solution = moderator_solution
            iteration_results["result"] = current_solution
            
            # Add iteration to results
            results["iterations"].append(iteration_results)
            
            # Check if moderator thinks we're done
            if "solution is complete" in moderator_solution.lower() or "no further iterations needed" in moderator_solution.lower():
                results["early_completion"] = True
                break
        
        # Set final result
        results["final_solution"] = current_solution
        
        return results
