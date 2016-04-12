import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.core.Distribution;
import bn.core.RandomVariable;
import bn.parser.BIFParser;
import bn.parser.XMLBIFParser;
import java.lang.Math;

/**
 * 
 * @author Trevor Whitestone
 *
 */
public class MyInferencer{
	BayesianNetwork bn;
	
	/*
	 * 1st element of argv should be file name
	 * 2nd should be the query variable
	 * Then put in all evidence variables followed immediately by their respective observed values
	 * 2nd to last should be boolean which, if true, makes likelihood-weighting be used or false for rejection sampling
	 * Last element should be the amount of trials to perform 
	 */
	public static void main(String[] argv) throws IOException
	{
		String fileName = argv[0];
		BayesianNetwork bn = null;
		XMLBIFParser parser = new XMLBIFParser();
		try {
			bn = parser.readNetworkFromFile(fileName);
		} catch (IOException | ParserConfigurationException | SAXException e) {
			e.printStackTrace();
			try {
				BIFParser bifParser = new BIFParser((new FileInputStream(fileName)));
				bn = bifParser.parseNetwork();
			} catch (FileNotFoundException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
		Assignment e = new Assignment();
		String query = argv[1];
		for (int i = 2; i < argv.length-2; i+= 2)
		{
			e.set(bn.getVariableByName(argv[i]), argv[i+1]);
		}
		MyInferencer en = new MyInferencer(bn);
		long start = System.nanoTime();
		Distribution d1 = en.ask(bn.getVariableByName(query), e);
		System.out.println("time elapsed " + (System.nanoTime() - start)/(1000000000.0));
		System.out.println("Exact Inference Distribution for " + query + ": " + d1);
		start = System.nanoTime();
		int amt = Integer.parseInt(argv[argv.length-1]);
		System.out.println("# trials: "+amt);
		Distribution d2 = null;
		String method = "rejection sampling";
		if (Boolean.parseBoolean(argv[argv.length-2]))
		{
			d2 = en.weighting(bn.getVariableByName(query), e,amt);
			method = "likelihood weighting";
		}
		else d2 = en.rejectionSample(bn.getVariableByName(query), e,amt);
		System.out.println("time elapsed " + (System.nanoTime() - start)/(1000000000.0));
		System.out.println("Approximate Distribution from " + method + " for " + query + ": " + d2);
		double relError = 0;
		for (Object cur : d2.keySet())
		{
			relError += 100*Math.abs(d1.get(cur)-d2.get(cur))/(d1.get(cur));
		}
		relError /= d2.size();
		System.out.println("Relative % error: " + relError);
	}
	
	public MyInferencer(BayesianNetwork bn) {
		this.bn = bn;
	}

	public Distribution ask(RandomVariable X, Assignment e) {
		Distribution q = new Distribution();
		for (Object each : X.getDomain())
		{
			Assignment temp = (Assignment) e.clone();
			temp.set(X, each);
			q.put(each,enumerate(bn.getVariableListTopologicallySorted(),temp));
		}
		q.normalize();
		return q;
	}
	
	private double enumerate(List<RandomVariable> vars, Assignment e)
	{
		if (vars.isEmpty()) return 1.0;
		RandomVariable Y = vars.remove(0);
		Boolean hasVal = false;
		Boolean val = null;
		try{
			if (e.get(Y)!=null) val = Boolean.parseBoolean((String)e.get(Y));
			if (val!=null)hasVal = true;
			
		}
		catch (NoSuchElementException n){
			
		}
//		System.out.println(Y + " " + val + " " + hasVal);
//		System.out.println(e);
//		System.out.println(vars);
		if (hasVal)
		{
			List<RandomVariable> temp = new ArrayList<>();
			temp.addAll(vars);
			
//			System.out.println("hasVal was true, "+bn.getProb(Y,e));
			return bn.getProb(Y,e)*enumerate(temp,e);
		}
		else
		{
			double sum = 0.0;
			for (Object each : Y.getDomain())
			{
				List<RandomVariable> temp = new ArrayList<>(); //Temporary list of Random Variables for use in recursion
				temp.addAll(vars);
				Assignment temp2 = (Assignment) e.clone(); //Temporary assignment of variables for use in recursion
				temp2.set(Y, each);
//				System.out.println("hasVal was false, "+bn.getProb(Y, temp2));
				sum +=  bn.getProb(Y, temp2) * enumerate(temp,temp2); // Implemented the updated Assignments because of the way that getProb works
			}
//			System.out.println(Y+" "+sum);
			return sum;
		}
	}
	
	
//	/*
//	 * Idea for gibbs: create a temp assignment variable with only markov blanket
//	 */
	private Object setRandomVal(RandomVariable X, Assignment e)
	{
//		System.out.println(X);
//		//System.out.println(e);
		Assignment temp = (Assignment) e.clone();
		double rand = Math.random();
		for (Object cur : X.getDomain())
		{
			temp.set(X, cur);
			double prob = bn.getProb(X, temp);
			if (rand <= prob) return cur;
			rand -= prob;
		}
		return null;
	}
	
	/*
	 * The latter must be the actual given variable assignments
	 */
	private boolean consistent(Assignment temp, Assignment act)
	{
		for (RandomVariable cur : act.keySet())
		{
			try {
				if (!((act.get(cur)+"").equals(temp.get(cur)+""))) return false;
			}
			catch (NoSuchElementException | NullPointerException e ) {
				return false;
			}
		}
		return true;
	}
	
	public Distribution rejectionSample(RandomVariable X, Assignment e, int n)
	{
		Distribution d = new Distribution();
		List<RandomVariable> vars = bn.getVariableListTopologicallySorted();
		for (Object cur : X.getDomain())
		{
			d.put(cur, 0);
		}
		//System.out.println(d);
		for (int i = 0; i < n; i++)
		{
			Assignment temp = (Assignment) e.clone();
			for (RandomVariable cur : vars)
			{
				temp.set(cur, setRandomVal(cur,temp)+"");
			}
			if (consistent(temp,e))
			{
				Object curValue = (temp.get(X)+""); //had to use string values for everything
				d.put(curValue, d.get(curValue)+1);
			}
		}
		// Check the % of actually generated values
		//System.out.println(d);
		d.normalize();
		return d;
	}
	
	public Distribution weighting(RandomVariable X, Assignment e, int n){
		Distribution d = new Distribution();
		for (Object cur: X.getDomain())
		{
			d.put(cur+"", 0.0);
		}
//		d.put("true", 0.0);
//		d.put("false", 0.0);
		for (int i = 0; i < n; i++)
		{
			Object[] retArray = weightedSample(e);
			Assignment x = (Assignment) retArray[0];
			double w = (double) retArray[1];
			d.put(x.get(X), (double)(d.get(x.get(X)))+w);
		}
		d.normalize();
		return d;
	}

	private Object[] weightedSample(Assignment e) {
		double w = 1;
		Assignment x = (Assignment) e.clone();
		for (RandomVariable cur : bn.getVariableListTopologicallySorted())
		{	
			Assignment temp = (Assignment) x.clone();
			temp.set(cur, "true");
			Boolean val = null;
			Boolean hasVal = false;
			try{
				if (e.get(cur)!=null) val = Boolean.parseBoolean((String)e.get(cur));
				if (val!=null){
					hasVal = true;
				}
				
			}
			catch (NoSuchElementException n){
				
			}
			if (hasVal)
			{
				w *= bn.getProb(cur, temp);
			}
			else x.put(cur,setRandomVal(cur,x)+""); //another case of the string problem
		}
		Object[] toReturn = new Object[2];
		toReturn[0] = x;
		toReturn[1] = w;
		return toReturn;
	}
	
}
