import React from 'react'
import TextField from '@material-ui/core/TextField'

const Params = (props) => {
  let items = props.params.map((param) => {
    if (param.type === 'string') {
      if ('choices' in param) {
        return <TextField
          select
          onChange={(e) => props.onChange(param.name, e.target.value)}
          label={param.name}
        >
          {param.choices.map(c => (<option key={c} value={c}>{c}</option>))}
        </TextField>
      }
      else {
        return <TextField
          key={param.name}
          required={param.required}
          label={param.name}
          onChange={(e) => props.onChange(param.name, e.target.value)}
        />
      }
    }
    else if (param.type === 'int') {
      return <TextField
        key={param.name}
        required={param.required}
        label={param.name}
        onChange={(e) => props.onChange(param.name, parseInt(e.target.value))}
        type='number'
      />
    }
  })

  return <div>{items}</div>
}

export default Params