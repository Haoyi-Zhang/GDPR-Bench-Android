"""
RAG (Retrieval-Augmented Generation) Method for GDPR Compliance Detection - FIXED VERSION

Fixed Issues:
1. ✅ Fixed dataset path issues (supports multiple path search)
2. ✅ Fixed vectorstore path issues
3. ✅ Added better error handling
4. ✅ Optimized knowledge base construction
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from methods.base_method import BaseMethod

# Import RAG components
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG components not available: {e}")
    RAG_AVAILABLE = False
    
    class Document:
        """Placeholder for langchain Document"""
        def __init__(self, page_content='', metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class RAGMethod(BaseMethod):
    """
    RAG-based method for GDPR compliance detection - FIXED VERSION
    
    Improvements:
    - Better file path handling
    - More robust error handling
    - Optimized knowledge base construction
    """
    
    def initialize(self):
        """Initialize the RAG system with GDPR knowledge base."""
        if not RAG_AVAILABLE:
            print("RAG components not available, using fallback mode")
            self.rag_chain = None
            return
        
        # Configuration
        self.model = self.config.get('model', 'gpt-4o')
        self.api_base = self.config.get('api_base', 'https://api.openai.com/v1')
        self.api_key = self.config.get('api_key')
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-ada-002')
        self.embedding_api_base = self.config.get('embedding_api_base', 'https://api.openai.com/v1')
        self.embedding_api_key = self.config.get('embedding_api_key')
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.retrieval_k = self.config.get('retrieval_k', 5)
        self.temperature = self.config.get('temperature', 0.0)
        
        # FIX 1: Set working directory and dataset path
        self.dataset_path = self._find_dataset_file()
        # Use independent vectorstore directory for different configs and threads to avoid concurrency conflicts
        import hashlib
        import threading
        import time
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        thread_id = threading.get_ident()
        timestamp = int(time.time() * 1000)
        default_dir = f'./gdpr_vectorstore_{config_hash}_t{thread_id}_{timestamp}'
        self.vectorstore_dir = self.config.get('vectorstore_dir', default_dir)
        
        print(f"Initializing RAG system...")
        print(f"Model: {self.model}")
        print(f"Embedding: {self.embedding_model}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Vectorstore: {self.vectorstore_dir}")
        
        # Build GDPR knowledge base
        try:
            self.vectorstore = self._build_gdpr_knowledge_base()
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retrieval_k}
            )
            
            # Build RAG chain
            self.rag_chain = self._build_rag_chain()
            print("✅ RAG system initialized successfully")
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize RAG system: {e}"
            print(error_msg)
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(error_msg, exc_info=True)
            traceback.print_exc()
            self.rag_chain = None
    
    def _find_dataset_file(self) -> str:
        """
        FIX: Intelligently find dataset file
        
        Try multiple possible paths
        """
        possible_paths = [
            'GDPR_dataset.json',                    # Current directory
            '../GDPR_dataset.json',                 # Parent directory
            '../../GDPR_dataset.json',              # Grandparent directory
            '/tmp/GDPR_dataset.json',               # /tmp directory
            os.path.expanduser('~/GDPR_dataset.json'),  # User home directory
        ]
        
        # Check if path is specified in config
        if self.config.get('dataset_path'):
            possible_paths.insert(0, self.config['dataset_path'])
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found dataset at: {path}")
                return path
        
        print("⚠️  Warning: GDPR_dataset.json not found, will use definitions only")
        return None
    
    def _build_gdpr_knowledge_base(self):
        """
        Build GDPR knowledge base from:
        1. GDPR article definitions
        2. Annotated examples from dataset (if available)
        """
        documents = []
        
        # 1. Add GDPR article definitions (always available)
        print("Loading GDPR article definitions...")
        gdpr_definitions = self._load_gdpr_definitions()
        for article_num, definition in gdpr_definitions.items():
            doc = Document(
                page_content=f"Article {article_num}: {definition}",
                metadata={
                    "source": "gdpr_articles",
                    "article": article_num
                }
            )
            documents.append(doc)
        print(f"✅ Loaded {len(documents)} GDPR article definitions")
        
        # 2. Add annotated examples from dataset (if available)
        if self.dataset_path:
            print(f"Loading annotated examples from {self.dataset_path}...")
            try:
                annotated_examples = self._load_annotated_examples()
                documents.extend(annotated_examples)
                print(f"✅ Loaded {len(annotated_examples)} annotated examples")
            except Exception as e:
                print(f"⚠️  Warning: Could not load annotated examples: {e}")
        
        print(f"Total documents: {len(documents)}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")
        
        # FIX 2: Ensure vectorstore directory exists
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Embed and store
        print("Creating embeddings...")
        try:
            embedding = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_base=self.embedding_api_base,
                openai_api_key=self.embedding_api_key
            )
        except Exception as e:
            print(f"⚠️ OpenAI Embeddings failed ({e}), trying HuggingFace fallback...")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("✅ Using HuggingFace Embeddings (offline)")
            except Exception as e2:
                print(f"❌ HuggingFace also failed ({e2})")
                raise RuntimeError("No embedding model available") from e
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=self.vectorstore_dir
        )
        
        print("✅ Vectorstore created")
        return vectorstore
    
    def _load_gdpr_definitions(self) -> Dict[int, str]:
        """Load comprehensive GDPR article definitions."""
        return {
            5: "Article 5: Principles relating to processing of personal data - Personal data must be: (a) processed lawfully, fairly and transparently; (b) collected for specified, explicit and legitimate purposes; (c) adequate, relevant and limited to what is necessary; (d) accurate and kept up to date; (e) kept in a form which permits identification for no longer than necessary; (f) processed in a manner that ensures appropriate security.",
            
            6: "Article 6: Lawfulness of processing - Processing is lawful only if at least one of the following applies: (a) the data subject has given consent; (b) processing is necessary for performance of a contract; (c) processing is necessary for compliance with a legal obligation; (d) processing is necessary to protect vital interests; (e) processing is necessary for performance of a task in public interest; (f) processing is necessary for legitimate interests pursued by controller or third party.",
            
            7: "Article 7: Conditions for consent - Where processing is based on consent: (a) controller must be able to demonstrate consent was given; (b) request for consent must be clearly distinguishable from other matters; (c) data subject has right to withdraw consent at any time; (d) consent must be as easy to withdraw as to give. When assessing whether consent is freely given, account shall be taken of whether performance of contract is conditional on consent.",
            
            8: "Article 8: Conditions applicable to child's consent - Where child is below age of 16 years, processing is lawful only if consent is given or authorized by holder of parental responsibility. Member States may provide by law for lower age (not below 13 years).",
            
            9: "Article 9: Processing of special categories of personal data - Processing of personal data revealing racial/ethnic origin, political opinions, religious/philosophical beliefs, trade union membership, genetic data, biometric data, health data, or data concerning sex life/sexual orientation is PROHIBITED unless: (a) explicit consent; (b) necessary for employment/social security law; (c) necessary to protect vital interests; (d) processing by foundation/association; (e) data manifestly made public; (f) necessary for legal claims; (g) necessary for substantial public interest; (h) necessary for health/social care; (i) necessary for public health; (j) necessary for archiving/research/statistics.",
            
            12: "Article 12: Transparent information, communication and modalities - Controller must take appropriate measures to provide any information referred to in Articles 13-14 and any communication under Articles 15-22 in concise, transparent, intelligible and easily accessible form, using clear and plain language. Information shall be provided in writing or by other means, including electronic means. Controller may provide information orally if requested. Information shall be provided free of charge.",
            
            13: "Article 13: Information to be provided where personal data are collected from the data subject - Where personal data are collected from data subject, controller must provide: (a) identity and contact details of controller; (b) contact details of data protection officer; (c) purposes and legal basis of processing; (d) legitimate interests where applicable; (e) recipients or categories of recipients; (f) intention to transfer to third country; (g) period for which data will be stored; (h) existence of rights (access, rectification, erasure, restriction, object, data portability); (i) right to withdraw consent; (j) right to lodge complaint; (k) whether provision of data is statutory/contractual requirement; (l) existence of automated decision-making including profiling.",
            
            14: "Article 14: Information to be provided where personal data have not been obtained from the data subject - Where personal data not obtained from data subject, controller must provide same information as Article 13, plus: (a) categories of personal data concerned; (b) source of personal data. Information must be provided within reasonable period (at latest one month), or at first communication, or when disclosure to another recipient is envisaged.",
            
            15: "Article 15: Right of access by the data subject - Data subject has right to obtain from controller confirmation as to whether personal data concerning them are being processed, and if so: (a) purposes of processing; (b) categories of data; (c) recipients; (d) retention period; (e) existence of rights; (f) right to lodge complaint; (g) information about source; (h) existence of automated decision-making. Controller must provide copy of personal data undergoing processing.",
            
            16: "Article 16: Right to rectification - Data subject has right to obtain from controller without undue delay rectification of inaccurate personal data and to have incomplete personal data completed.",
            
            17: "Article 17: Right to erasure ('right to be forgotten') - Data subject has right to obtain erasure of personal data without undue delay where: (a) data no longer necessary; (b) data subject withdraws consent; (c) data subject objects and no overriding grounds; (d) data unlawfully processed; (e) erasure required for compliance with legal obligation; (f) data collected in relation to information society services to children.",
            
            18: "Article 18: Right to restriction of processing - Data subject has right to obtain restriction where: (a) accuracy of data is contested; (b) processing is unlawful but data subject opposes erasure; (c) controller no longer needs data but subject needs it for legal claims; (d) data subject has objected to processing pending verification.",
            
            19: "Article 19: Notification obligation regarding rectification or erasure - Controller must communicate any rectification, erasure, or restriction to each recipient to whom data have been disclosed, unless impossible or involves disproportionate effort. Controller must inform data subject about those recipients if requested.",
            
            20: "Article 20: Right to data portability - Data subject has right to receive personal data in structured, commonly used and machine-readable format and have right to transmit those data to another controller where: (a) processing is based on consent or contract; (b) processing is carried out by automated means.",
            
            21: "Article 21: Right to object - Data subject has right to object to processing based on legitimate interests or for performance of task in public interest. Controller must no longer process unless it demonstrates compelling legitimate grounds. Data subject has absolute right to object to processing for direct marketing purposes.",
            
            25: "Article 25: Data protection by design and by default - Controller must implement appropriate technical and organizational measures (e.g., pseudonymization) to implement data-protection principles and safeguard rights. Controller must implement measures to ensure that by default only personal data necessary for each specific purpose are processed (amount collected, extent of processing, period of storage, accessibility). Measures must take into account state of art, cost, nature/scope/context/purposes of processing, and risks.",
            
            30: "Article 30: Records of processing activities - Each controller must maintain record of processing activities under its responsibility containing: (a) name and contact details of controller; (b) purposes of processing; (c) description of categories of data subjects and data; (d) categories of recipients; (e) transfers to third countries; (f) retention periods; (g) description of technical and organizational security measures. Each processor must maintain record containing: (a) name and contact details; (b) categories of processing carried out on behalf of each controller; (c) transfers to third countries; (d) description of security measures.",
            
            32: "Article 32: Security of processing - Controller and processor must implement appropriate technical and organizational measures to ensure level of security appropriate to risk, including: (a) pseudonymization and encryption of personal data; (b) ability to ensure ongoing confidentiality, integrity, availability and resilience of processing systems; (c) ability to restore availability and access to data in timely manner after incident; (d) process for regularly testing, assessing and evaluating effectiveness of measures. In assessing appropriate level of security, account must be taken of: state of art, costs of implementation, nature/scope/context/purposes of processing, and risk of varying likelihood and severity for rights and freedoms of natural persons.",
            
            33: "Article 33: Notification of a personal data breach to the supervisory authority - In case of personal data breach, controller must without undue delay and where feasible not later than 72 hours after becoming aware, notify breach to supervisory authority unless unlikely to result in risk to rights and freedoms. Notification must describe: (a) nature of breach; (b) name and contact details of data protection officer; (c) likely consequences; (d) measures taken or proposed to address breach.",
            
            35: "Article 35: Data protection impact assessment - Where type of processing (particularly using new technologies) is likely to result in high risk to rights and freedoms, controller must carry out assessment of impact of processing operations on protection of personal data. Assessment must contain: (a) systematic description of processing operations and purposes; (b) assessment of necessity and proportionality; (c) assessment of risks; (d) measures envisaged to address risks.",
            
            44: "Article 44: General principle for transfers - Any transfer of personal data to third country or international organization may take place only if: controller and processor comply with conditions laid down in this Chapter, and other provisions of GDPR apply. All provisions of this Chapter must be applied to ensure level of protection of natural persons guaranteed by GDPR is not undermined.",
            
            46: "Article 46: Transfers subject to appropriate safeguards - Transfer may take place where controller or processor has provided appropriate safeguards and enforceable data subject rights are available: (a) legally binding instrument between public authorities; (b) binding corporate rules; (c) standard data protection clauses adopted by Commission; (d) standard data protection clauses adopted by supervisory authority; (e) approved code of conduct; (f) approved certification mechanism.",
            
            58: "Article 58: Powers - Each supervisory authority has: (a) investigative powers (order controller/processor to provide information, carry out investigations, obtain access to premises); (b) corrective powers (issue warnings, reprimands, order compliance, order limitation/ban on processing, order rectification/erasure, impose administrative fines); (c) authorization and advisory powers (advise controller, issue opinions, authorize contractual clauses).",
            
            83: "Article 83: General conditions for imposing administrative fines - When deciding whether to impose fine and amount, due regard must be given to: (a) nature, gravity and duration of infringement; (b) intentional or negligent character; (c) action taken to mitigate damage; (d) degree of responsibility; (e) relevant previous infringements; (f) degree of cooperation with authority; (g) categories of data affected; (h) manner in which infringement became known; (i) other aggravating or mitigating factors. Maximum fines: up to 10M EUR or 2% of annual worldwide turnover (whichever higher) for certain infringements; up to 20M EUR or 4% of annual worldwide turnover (whichever higher) for more serious infringements.",
            
            # ==================== Extended Articles (AI Era + Security Response) ====================
            
            4: "Article 4: Definitions - Key terms: 'personal data' means any information relating to identified or identifiable natural person ('data subject'); 'processing' means any operation or set of operations performed on personal data (collection, recording, organization, structuring, storage, adaptation, retrieval, consultation, use, disclosure, dissemination, restriction, erasure, destruction); 'controller' means natural or legal person determining purposes and means of processing; 'processor' means natural or legal person processing personal data on behalf of controller; 'consent' means any freely given, specific, informed and unambiguous indication of data subject's wishes by statement or clear affirmative action; 'personal data breach' means breach of security leading to accidental or unlawful destruction, loss, alteration, unauthorized disclosure of, or access to, personal data.",
            
            22: "Article 22: Automated individual decision-making, including profiling - Data subject has right not to be subject to decision based solely on automated processing, including profiling, which produces legal effects concerning them or similarly significantly affects them. This does not apply if decision: (a) is necessary for entering into or performance of contract between data subject and controller; (b) is authorized by Union or Member State law; (c) is based on data subject's explicit consent. In cases (a), (b), and (c), controller must implement suitable measures to safeguard data subject's rights and freedoms and legitimate interests, at least right to obtain human intervention on part of controller, to express point of view and to contest decision. Automated decision-making based on special categories of personal data requires explicit consent or substantial public interest with suitable safeguards.",
            
            24: "Article 24: Responsibility of the controller - Taking into account nature, scope, context and purposes of processing as well as risks of varying likelihood and severity for rights and freedoms of natural persons, controller must implement appropriate technical and organizational measures to ensure and to be able to demonstrate that processing is performed in accordance with GDPR ('accountability'). Measures must be reviewed and updated where necessary. Where proportionate in relation to processing activities, measures must include implementation of appropriate data protection policies by controller. Adherence to approved codes of conduct or certification mechanisms may be used to demonstrate compliance.",
            
            28: "Article 28: Processor - Where processing is to be carried out on behalf of controller, controller must use only processors providing sufficient guarantees to implement appropriate technical and organizational measures in manner that processing meets requirements of GDPR and ensures protection of rights. Processor must not engage another processor without prior specific or general written authorization of controller. Processing by processor must be governed by contract or other legal act binding processor to controller and stipulating: (a) subject-matter and duration of processing; (b) nature and purpose of processing; (c) type of personal data and categories of data subjects; (d) obligations and rights of controller. Contract must stipulate that processor: (a) processes data only on documented instructions from controller; (b) ensures confidentiality of persons authorized to process; (c) takes measures required pursuant to Article 32; (d) respects conditions for engaging another processor; (e) assists controller in responding to requests for exercising rights; (f) assists controller in ensuring compliance with Articles 32-36; (g) deletes or returns all personal data after end of provision of services; (h) makes available to controller all information necessary to demonstrate compliance.",
            
            34: "Article 34: Communication of personal data breach to data subject - When personal data breach is likely to result in high risk to rights and freedoms of natural persons, controller must communicate breach to data subject without undue delay. Communication must be in clear and plain language and describe: (a) nature of personal data breach; (b) name and contact details of data protection officer or other contact point; (c) likely consequences of breach; (d) measures taken or proposed by controller to address breach and mitigate possible adverse effects. Communication not required if controller: (a) has implemented appropriate technical and organizational protection measures (particularly encryption) that render data unintelligible to unauthorized persons; (b) has taken subsequent measures ensuring high risk to rights and freedoms is no longer likely to materialize; (c) it would involve disproportionate effort (in which case public communication or similar measure whereby data subjects are informed in equally effective manner instead)."
        }
    
    def _load_annotated_examples(self, limit=100) -> List[Document]:
        """
        Load annotated code examples from GDPR_dataset.json.
        
        ✅ FIX: Better error handling
        """
        documents = []
        
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return documents
        
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            # Limit to avoid too many examples
            dataset = dataset[:limit]
            
            for item in dataset:
                code_snippet = item.get("code_snippet", "")
                if isinstance(code_snippet, list):
                    code_snippet = "\n".join(code_snippet)
                
                violated_article = item.get("violated_article", 0)
                violation_type = item.get("violation_type", "")
                annotation_note = item.get("annotation_note", "")
                
                # Create a document from the annotated example
                content = f"""GDPR Violation Example (Article {violated_article}):

Violation Type: {violation_type}

Code Snippet:
{code_snippet[:500]}

Explanation: {annotation_note[:200]}

This code violates GDPR Article {violated_article} because it {violation_type.lower()}."""
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "annotated_examples",
                        "article": violated_article,
                        "violation_type": violation_type,
                        "app": item.get("app_name", "")
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            print(f"⚠️  Warning: Could not load annotated examples: {e}")
        
        return documents
    
    def _build_rag_chain(self):
        """Build RAG chain with improved prompt."""
        
        # LLM
        llm = ChatOpenAI(
            model_name=self.model,
            temperature=self.temperature,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key
        )
        
        # Format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Improved prompt
        prompt = ChatPromptTemplate.from_template("""You are a GDPR compliance expert. Analyze the code and determine which GDPR articles are violated.

**Relevant GDPR Knowledge:**
{context}

**Code to Analyze:**
{question}

**Instructions:**
1. Carefully review the GDPR knowledge above
2. Analyze the code for privacy violations
3. Identify ALL violated GDPR article numbers

**Output Format:**
Provide ONLY the violated article numbers as comma-separated integers.
Examples:
- If violations found: "6,7,32"
- If no violations: "0"

Do NOT include explanations, just the numbers.

Answer:""")
        
        # RAG Chain
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _extract_articles_from_response(self, response: str) -> List[int]:
        """Extract article numbers from RAG response."""
        response = response.strip()
        
        # Try direct number extraction
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            articles = [int(n) for n in numbers if n.isdigit() and int(n) <= 99]
            return articles if articles else [0]
        
        return [0]
    
    def _run_rag_query(self, code: str) -> List[int]:
        """Run RAG query on code snippet."""
        if not self.rag_chain:
            return self._fallback_analysis(code)
        
        try:
            # Invoke RAG chain
            result = self.rag_chain.invoke(code)
            
            # Extract articles
            articles = self._extract_articles_from_response(result)
            
            print(f"[RAG Response]: {result[:200]}...")
            print(f"[Extracted Articles]: {articles}")
            
            return articles
            
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(code)
    
    def _fallback_analysis(self, code: str) -> List[int]:
        """Simple fallback if RAG fails."""
        articles = set()
        
        # Basic pattern matching
        if re.search(r'getDeviceId|IMEI|ANDROID_ID', code, re.IGNORECASE):
            articles.update([6, 13])
        if re.search(r'Camera\.open|MediaRecorder', code, re.IGNORECASE):
            articles.update([6, 7, 32])
        if re.search(r'getLocation|LocationManager', code, re.IGNORECASE):
            articles.update([6, 9])
        if re.search(r'http://', code, re.IGNORECASE):
            articles.add(32)
        if re.search(r'sendTextMessage|SmsManager', code, re.IGNORECASE):
            articles.update([6, 13])
        
        return sorted(list(articles)) if articles else [0]
    
    def predict_file_level(self, file_path: str, code: str, **kwargs) -> List[int]:
        """Analyze file using RAG."""
        query = f"File: {file_path}\n\nCode:\n{code[:2000]}"
        return self._run_rag_query(query)
    
    def predict_module_level(self, file_path: str, module_name: str, 
                            code: str, **kwargs) -> List[int]:
        """Analyze module using RAG."""
        query = f"Module: {module_name} in {file_path}\n\nCode:\n{code[:2000]}"
        return self._run_rag_query(query)
    
    def predict_line_level(self, file_path: str, line_spans: str, 
                          code: str, description: str, **kwargs) -> List[int]:
        """Analyze code lines using RAG."""
        query = f"Lines {line_spans} in {file_path}\nDescription: {description}\n\nCode:\n{code}"
        return self._run_rag_query(query)
    
    def predict_snippet(self, snippet: str, snippet_path: str = "", **kwargs) -> List[int]:
        """Analyze code snippet using RAG."""
        query = f"Code snippet from {snippet_path}:\n\n{snippet}"
        return self._run_rag_query(query)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'vectorstore') and self.vectorstore:
            try:
                # Vectorstore will automatically persist to disk
                pass
            except:
                pass
